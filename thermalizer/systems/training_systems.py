import os
import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import thermalizer.dataset.datasets as datasets
import thermalizer.models.misc as misc
import thermalizer.models.diffusion as diffusion
import thermalizer.kolmogorov.performance as performance
import thermalizer.qg.performance as performance_qg
import thermalizer.models.refiner as refiner


def setup():
    """Sets up the process group for distributed training.
       We are using torchrun so not using rank and world size arguments """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

def cleanup():
    """Cleans up the process group."""
    dist.destroy_process_group()

def trainer_from_checkpoint(checkpoint_string):
    with open(checkpoint_string, 'rb') as fp:
        model_dict = pickle.load(fp)
    if model_dict["config"]["model_type"]=='ModernUnetRegressor':
        trainer=ThermalizerTrainer(model_dict["config"])
    else:
        trainer=ResidualEmulatorTrainer(model_dict["config"])
    trainer.load_checkpoint(checkpoint_string)
    return trainer

class Trainer:
    """ Base trainer class """
    def __init__(self,config):
        self.config=config
        self.epoch=1 ## Initialise at first epoch
        self.training_step=0 ## Counter to keep track of number of weight updates
        self.wandb_init=False ## Bool to track whether or not wandb run has been initialised
        self.ema=None
        if self.config.get("wandb_log_freq"):
            self.log_freq=self.config.get("wandb_log_freq")
        else:
            self.log_freq=50

        self.gradient_clip=self.config["optimization"].get("gradient_clip")
        
        if self.config["ddp"]:
            setup()
            self.gpu_id=int(os.environ["LOCAL_RANK"])
            self.ddp=True
            self.world_size=dist.get_world_size()
            self.config["world_size"]=self.world_size
            if self.gpu_id==0:
                self.logging=True
            else:
                self.logging=False
        else:
            self.gpu_id="cuda"
            self.ddp=False
            self.logging=True

        ## Leave these print statements for now during dev
        print("Prep data")
        self._prep_data()
        print("Prep model")
        self._prep_model()
        print("Prep optimizer")
        self._prep_optimizer()

    def init_wandb(self):
        ## Set up wandb stuff
        wandb.init(entity="qiliu2221",project=self.config["project"],
                        dir="/scratch/ql2221/thermalizer_data/wandb_data",config=self.config)
        self.config["save_path"]=wandb.run.dir
        self.config["wandb_url"]=wandb.run.get_url()
        self.wandb_init=True 
        ## Sync all configs
        wandb.config.update(self.config)
        self.model.config=self.config

    def resume_wandb(self):
        """ Resume a wandb run from the self.config wandb url. """
        wandb.init(entity="qiliu2221",project=self.config["project"],
                            id=self.config["wandb_url"][-8:],dir="/scratch/ql2221/thermalizer_data/wandb_data", resume="must")
        self.wandb_init=True
        return

    def _prep_data(self):
        if self.config["PDE"]=="Kolmogorov":
            train_data,valid_data,config=datasets.parse_data_file(self.config)
        elif self.config["PDE"]=="QG":
            train_data,valid_data,config=datasets.parse_data_file_qg(self.config)
        else:
            print("Need to know what PDE system we are working with")
            quit()

        ds_train=datasets.FluidDataset(train_data)
        ds_valid=datasets.FluidDataset(valid_data)
        self.config=config ## Update config dict

        if self.ddp:
            train_sampler=DistributedSampler(ds_train)
            valid_sampler=DistributedSampler(ds_valid)
        else:
            train_sampler=RandomSampler(ds_train)
            valid_sampler=RandomSampler(ds_valid)

        self.train_loader=DataLoader(
                ds_train,
                num_workers=self.config["loader_workers"],
                batch_size=self.config["optimization"]["batch_size"],
                sampler=train_sampler,
            )

        self.valid_loader=DataLoader(
                ds_valid,
                num_workers=self.config["loader_workers"],
                batch_size=self.config["optimization"]["batch_size"],
                sampler=valid_sampler,
            )

    def _prep_model(self):
        self.model=misc.model_factory(self.config).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())

        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()

        if self.ddp:
            self.model = DDP(self.model,device_ids=[self.gpu_id])

    def _prep_optimizer(self):
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),
                            lr=self.config["optimization"]["lr"],
                            weight_decay=self.config["optimization"]["wd"])
        if self.config["optimization"].get("scheduler_step"):
            self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,
                    self.config["optimization"]["scheduler_step"],
                    gamma=self.config["optimization"]["scheduler_gamma"], last_epoch=-1)
        else:
            self.scheduler=None

    def training_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def valid_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def run(self):
        raise NotImplementedError("Implemented by subclass")


class ResidualEmulatorTrainer(Trainer):
    """ Residual residual emulator - emulates residuals
        with loss defined on residuals at each timestep """
    def __init__(self,config):
        super().__init__(config)
        self.val_loss=0
        self.val_loss_check=100
        self.n_rollout=1
        self.config["emulator_loss"]="ResidualResidual"
        self.residual=True
        self.sigma=self.config.get("sigma")

    def training_loop(self):
        """ Training loop for residual emulator. We push loss to wandb
            after each batch/weight update """

        self.model.train()
        for x_data in self.train_loader:
            x_data=x_data.to(self.gpu_id)
            self.optimizer.zero_grad()
            loss=0
            state_loss=0
            if self.config["input_channels"]==1:
                x_data=x_data.unsqueeze(2)
            for aa in range(0,self.n_rollout):
                if aa==0:
                    x_t=x_data[:,0]
                else:
                    x_t=x_dt+x_t
                    if self.sigma:
                        x_t+=self.sigma*torch.randn_like(x_t,device=x_t.device)
                x_dt=self.model(x_t)
                loss_dt=self.criterion(x_dt,x_data[:,aa+1]-x_data[:,aa,:])
                loss_s=self.criterion(x_data[:,aa+1],x_t+x_dt)
                state_loss+=loss_s.item() ## Don't need gradient here
                loss+=loss_dt
            loss.backward()

            ## Gradient clipping if its a thing
            if self.gradient_clip:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.gradient_clip)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss_resid"]=loss.item()
                log_dic["train_loss_state"]=state_loss
                log_dic["training_step"]=self.training_step
                log_dic["n_rollout"]=self.n_rollout
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1

            if self.ema:
                self.ema.update()

            ## Check rollout scheduler
            if (self.training_step%self.config["rollout_scheduler"]==0) and self.n_rollout<self.config["max_rollout"]:
                self.n_rollout+=1


        return loss

    def valid_loop(self):
        """ Training loop for residual emulator. Aggregate loss over validation set for wandb update """
        log_dic={}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss=0
        epoch_loss_state=0
        nsamp=0
        with torch.no_grad():
            for x_data in self.valid_loader:
                x_data=x_data.to(self.gpu_id)
                nsamp+=x_data.shape[0]
                loss=0
                state_loss=0
                if self.config["input_channels"]==1:
                    x_data=x_data.unsqueeze(2)
                for aa in range(0,self.n_rollout):
                    if aa==0:
                        x_t=x_data[:,0]
                    else:
                        x_t=x_dt+x_t
                        if self.sigma:
                            x_t+=self.sigma*torch.randn_like(x_t,device=x_t.device)
                    x_dt=self.model(x_t)
                    loss_dt=self.criterion(x_dt,x_data[:,aa+1]-x_data[:,aa,:])
                    loss_s=self.criterion(x_data[:,aa+1],x_t+x_dt)
                    loss+=loss_dt
                    state_loss+=loss_s
                epoch_loss+=loss.detach()*x_data.shape[0]
                epoch_loss_state+=state_loss.detach()*x_data.shape[0]
        epoch_loss/=nsamp ## Average over full epoch
        epoch_loss_state/=nsamp 
        ## Now we want to allreduce loss over all processes
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_loss_state, op=dist.ReduceOp.SUM)
            ## Average across all processes
            self.val_loss = epoch_loss.item()/self.world_size
            epoch_loss_state=epoch_loss_state/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss_resid"]=self.val_loss ## Average over full epoch
            log_dic["valid_loss_state"]=epoch_loss_state.item()
            log_dic["training_step"]=self.training_step
            log_dic["n_rollout"]=self.n_rollout
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return loss

    def checkpointing(self):
        """ Checkpointing performs two actions:
            1. Save last checkpoint.
            2. Overwrite lowest validation checkpoint if val loss is lower
        """
        if self.epoch==1:
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_best.p")
        if self.ema:
            self.ema.apply_shadow()
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_best_ema.p")
            self.ema.restore()

        self.save_checkpoint(self.config["save_path"]+"/checkpoint_last.p")
        if self.ema:
            self.ema.apply_shadow()
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_last_ema.p")
            self.ema.restore()
        if (self.epoch>1) and (self.val_loss<self.val_loss_check) and self.n_rollout==self.config["max_rollout"]:
            print("Saving new checkpoint with improved validation loss at %s" % self.config["save_path"]+"/checkpoint_best.p")
            self.val_loss_check=self.val_loss ## Update checkpointed validation loss
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_best.p")
            if self.ema:
                self.ema.apply_shadow()
                self.save_checkpoint(self.config["save_path"]+"/checkpoint_best_ema.p")
                self.ema.restore()
        return
        
    def save_checkpoint(self, checkpoint_string):
        """ Checkpoint model and optimizer """

        if self.ddp:
            state_dict_buffer=self.model.module.state_dict()
        else:
            state_dict_buffer=self.model.state_dict()

        save_dict={
                    'epoch': self.epoch,
                    'training_step': self.training_step,
                    'state_dict': state_dict_buffer,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.val_loss,
                    'config':self.config,
                    }
        with open(checkpoint_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_checkpoint(self,file_string):
        """ Load checkpoint from saved file """
        with open(file_string, 'rb') as fp:
            model_dict = pickle.load(fp)
        assert model_dict["config"]==self.config, "Configs not the same"
        self.model=misc.load_model(file_string).to(self.gpu_id)
        self.epoch=model_dict["epoch"]
        self.training_step=model_dict["training_step"]
        self._prep_optimizer()
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])

        self.n_rollout = model_dict.get("n_rollout") if model_dict.get("n_rollout") else 4

        if self.wandb_init==False:
            self.resume_wandb()

        return

    def run(self,epochs=None):
        if self.logging and self.wandb_init==False:
            self.init_wandb()

        if epochs:
            max_epochs=epochs
        else:
            max_epochs=self.config["optimization"]["epochs"]

        for epoch in range(self.epoch,max_epochs+1):
            self.epoch=epoch
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)
            self.training_loop()
            self.valid_loop()
            if self.logging: ## Only run checkpointing on logging node
                self.checkpointing()
        if self.ddp:
            cleanup()
        print("DONE on rank", self.gpu_id)

        return

    def performance_long(self,steps=int(1e5)):
        """ Run long run performance test - here we just run the emulator
            for a long time, without true simulation steps to compare against
            due to speed and memory limitations """

        if self.ema:
            self.ema.apply_shadow()

        ## Load test data
        with open("/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/test40.p", 'rb') as fp:
            test_suite = pickle.load(fp)

        ## Make sure train and test increments are the same
        assert test_suite["increment"]==self.config["increment"]

        fig_ens,fig_field=performance.long_run_figures(self.model,test_suite["data"][:,0,:,:].to("cuda")/self.model.config["field_std"],steps=steps,
                                    residual=self.residual,sigma=self.sigma)
        if self.logging and self.wandb_init:
            wandb.log({"Long Ens": wandb.Image(fig_ens)})
            wandb.log({"Long field": wandb.Image(fig_field)})
        plt.close()
        return 

    def performance(self,silence=True):
        """ Run some performance metrics """
        if self.ddp:
            self.model=self.model.module

        ## This is a really gross hacky thing for now, but I gotta run these jobs tonight
        ## In future we should split this performance module off into another object
        ## that can distinguish between Kolmogorov and QG emulators
        if self.config["PDE"]=="Kolmogorov":
            ## Load test data
            with open("/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/test40.p", 'rb') as fp:
                test_suite = pickle.load(fp)

            ## Make sure train and test increments are the same
            assert test_suite["increment"]==self.config["increment"]

            """ ## Commenting out while developing as it takes ages
            fig_ens,fig_field=performance.long_run_figures(self.model,test_suite["data"][:,0,:,:].to("cuda")/self.model.config["field_std"],steps=int(1e5),residual=self.residual))
            wandb.log({"Long Ens": wandb.Image(fig_ens)})
            wandb.log({"Long field": wandb.Image(fig_field)})
            plt.close()
            """

            ## Run rollout against test sims, plot MSE
            emu_rollout=performance.EmulatorRollout(test_suite["data"],self.model,residual=self.residual,sigma=self.sigma,silence=silence)
            emu_rollout.evolve()
            fig_mse=plt.figure(figsize=(14,5))
            plt.suptitle("MSE wrt. true trajectory, emu step=%.2f" % test_suite["increment"])
            plt.subplot(1,2,1)
            plt.plot(emu_rollout.mse_emu[0],color="blue",alpha=0.1,label="Emulator")
            plt.subplot(1,2,2)
            plt.loglog(emu_rollout.mse_emu[0],color="blue",alpha=0.1,label="Emulator")
            for aa in range(1,len(emu_rollout.mse_auto)):
                plt.subplot(1,2,1)
                plt.plot(emu_rollout.mse_emu[aa],color="blue",alpha=0.1)
                plt.subplot(1,2,2)
                plt.loglog(emu_rollout.mse_emu[aa],color="blue",alpha=0.1)
            plt.subplot(1,2,1)
            plt.yscale("log")
            plt.ylim(1e-3,1e5)
            plt.legend()
            plt.ylabel("MSE")
            plt.xlabel("# of steps")
            plt.subplot(1,2,2)
            plt.ylim(1e-3,1e5)
            plt.xlabel("# of steps")
            wandb.log({"Rollout MSE": wandb.Image(fig_mse)})
            plt.close()

            ## Plot random samples
            samps=6
            indices=np.random.randint(0,len(emu_rollout.test_suite),size=samps)
            time_snaps=np.random.randint(0,len(emu_rollout.test_suite[0]),size=samps)
            fig_samps=plt.figure(figsize=(20,6))
            plt.suptitle("Sim (top) and emu (bottom) at random time samples")
            for aa in range(samps):
                plt.subplot(2,samps,aa+1)
                plt.title("emu step # %d" % time_snaps[aa])
                plt.imshow(emu_rollout.test_suite[indices[aa],time_snaps[aa]],cmap=sns.cm.icefire,interpolation='none')
                plt.colorbar()

                plt.subplot(2,samps,aa+1+samps)
                plt.imshow(emu_rollout.emu[indices[aa],time_snaps[aa]],cmap=sns.cm.icefire,interpolation='none')
                plt.colorbar()
            plt.tight_layout()
            wandb.log({"Random samples": wandb.Image(fig_samps)})
            plt.close()

            ## Enstrophy figure, along true trajectory
            ens_fig=plt.figure(figsize=(12,5))
            plt.suptitle("Enstrophy over time")
            for aa in range(len(emu_rollout.test_suite)):
                plt.subplot(1,2,1)
                ens_sim=torch.sum(torch.abs(emu_rollout.test_suite[aa])**2,axis=(-1,-2))
                ens_emu=torch.sum(torch.abs(emu_rollout.emu[aa])**2,axis=(-1,-2))
                plt.plot(ens_sim,color="blue",alpha=0.2)
                plt.plot(ens_emu,color="red",alpha=0.2)
                plt.xlabel("Emulator passes")
                plt.ylim(0,15000)
                
                plt.subplot(1,2,2)
                plt.plot(ens_sim,color="blue",alpha=0.2)
                plt.plot(ens_emu,color="red",alpha=0.2)
                plt.xscale("log")
                plt.xlabel("Emulator passes")
                plt.ylim(0,15000)
            wandb.log({"Enstrophy": wandb.Image(ens_fig)})
            plt.close()

        else:
            ## Otherwise do QG rollout
            test_suite=torch.load("/scratch/cp3759/thermalizer_data/qg/test_eddy/eddy_dt5_20.pt") ## Saved normalised?
            ## Cut test suite based on nsteps
            emu_rollout=performance_qg.EmulatorRollout(test_suite.to("cuda"),self.model,sigma=self.sigma,silence=silence)
            emu_rollout.evolve()
            fig_mse=plt.figure(figsize=(14,5))
            plt.suptitle("MSE wrt. true trajectory")
            plt.subplot(1,2,1)
            plt.plot(emu_rollout.mse_emu[0],color="blue",alpha=0.1,label="Emulator")
            plt.subplot(1,2,2)
            plt.loglog(emu_rollout.mse_emu[0],color="blue",alpha=0.1,label="Emulator")
            for aa in range(1,len(emu_rollout.mse_auto)):
                plt.subplot(1,2,1)
                plt.plot(emu_rollout.mse_emu[aa],color="blue",alpha=0.1)
                plt.subplot(1,2,2)
                plt.loglog(emu_rollout.mse_emu[aa],color="blue",alpha=0.1)
            plt.subplot(1,2,1)
            plt.yscale("log")
            plt.ylim(1e-3,1e5)
            plt.legend()
            plt.ylabel("MSE")
            plt.xlabel("# of steps")
            plt.subplot(1,2,2)
            plt.ylim(1e-3,1e5)
            plt.xlabel("# of steps")
            wandb.log({"Rollout MSE": wandb.Image(fig_mse)})
            plt.close()

            fig_fields=plt.figure(figsize=(13,4.2))
            plt.suptitle("Emulated fields after %d steps" % len(emu_rollout.mse_emu[0]))
            for aa in range(1,11):
                plt.subplot(2,5,aa)
                plt.imshow(emu_rollout.emu[aa][-1][0].cpu(),cmap=cmocean.cm.balance)
                plt.colorbar()
            plt.tight_layout()
            wandb.log({"Rollout MSE": wandb.Image(fig_mse)})
            plt.close()

        return


class ResidualStateEmulatorTrainer(ResidualEmulatorTrainer):
    """ Residual state emulator - emulates residuals
        but with loss defined on state """
    def __init__(self,config):
        super().__init__(config)
        self.config["emulator_loss"]="ResidualState"

    def training_loop(self):
        """ Override residual emulator training loop with state loss function """

        self.model.train()
        for x_data in self.train_loader:
            x_data=x_data.to(self.gpu_id)
            self.optimizer.zero_grad()
            loss=0
            if self.config["input_channels"]==1:
                x_data=x_data.unsqueeze(2)
            for aa in range(0,self.n_rollout):
                if aa==0:
                    x_t=x_data[:,0]
                else:
                    x_t=x_dt+x_t
                x_dt=self.model(x_t)
                loss_dt=self.criterion(x_dt+x_t,x_data[:,aa+1])
                loss+=loss_dt
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss_state"]=loss.item()
                log_dic["training_step"]=self.training_step
                log_dic["n_rollout"]=self.n_rollout
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1

            ## Check rollout scheduler
            if (self.training_step%self.config["rollout_scheduler"]==0) and self.n_rollout<self.config["max_rollout"]:
                self.n_rollout+=1

        return loss

    def valid_loop(self):
        """ Training loop for residual emulator. Aggregate loss over validation set for wandb update """
        log_dic={}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss=0
        nsamp=0
        with torch.no_grad():
            for x_data in self.valid_loader:
                x_data=x_data.to(self.gpu_id)
                nsamp+=x_data.shape[0]
                loss=0
                if self.config["input_channels"]==1:
                    x_data=x_data.unsqueeze(2)
                for aa in range(0,self.n_rollout):
                    if aa==0:
                        x_t=x_data[:,0]
                    else:
                        x_t=x_dt+x_t
                    x_dt=self.model(x_t)
                    loss_dt=self.criterion(x_dt+x_t,x_data[:,aa+1])
                    loss+=loss_dt
                epoch_loss+=loss.detach()*x_data.shape[0]
        epoch_loss/=nsamp ## Average over full epoch
        ## Now we want to allreduce loss over all processes
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            ## Acerage across all processes
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss_state"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            log_dic["n_rollout"]=self.n_rollout
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
            
        return loss


class StateEmulatorTrainer(ResidualEmulatorTrainer):
    """ Residual state emulator - emulates residuals
        but with loss defined on state """
    def __init__(self,config):
        super().__init__(config)
        self.config["emulator_loss"]="ResidualState"
        self.residual=False ## Bool to pass to performance modules

    def training_loop(self):
        """ Training loop for residual emulator. We push loss to wandb
            after each batch/weight update """

        self.model.train()
        for x_data in self.train_loader:
            x_data=x_data.to(self.gpu_id)
            self.optimizer.zero_grad()
            loss=0
            if self.config["input_channels"]==1:
                x_data=x_data.unsqueeze(2)
            for aa in range(0,self.n_rollout):
                if aa==0:
                    x_t=x_data[:,0]
                x_t=self.model(x_t)
                loss_dt=self.criterion(x_t,x_data[:,aa+1])
                loss+=loss_dt
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss_state"]=loss.item()
                log_dic["training_step"]=self.training_step
                log_dic["n_rollout"]=self.n_rollout
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1

            ## Check rollout scheduler
            if (self.training_step%self.config["rollout_scheduler"]==0) and self.n_rollout<self.config["max_rollout"]:
                self.n_rollout+=1

            if self.ema:
                self.ema.update()

        return loss

    def valid_loop(self):
        """ Training loop for residual emulator. Aggregate loss over validation set for wandb update """
        log_dic={}
        self.model.eval()
        epoch_loss=0
        nsamp=0
        with torch.no_grad():
            for x_data in self.valid_loader:
                x_data=x_data.to(self.gpu_id)
                nsamp+=x_data.shape[0]
                loss=0
                if self.config["input_channels"]==1:
                    x_data=x_data.unsqueeze(2)
                for aa in range(0,self.n_rollout):
                    if aa==0:
                        x_t=x_data[:,0]
                    x_t=self.model(x_t)
                    loss_dt=self.criterion(x_t,x_data[:,aa+1])
                    loss+=loss_dt
                epoch_loss+=loss.detach()*x_data.shape[0]
        epoch_loss/=nsamp ## Average over full epoch
        ## Now we want to allreduce loss over all processes
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            ## Acerage across all processes
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss_state"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            log_dic["n_rollout"]=self.n_rollout
            wandb.log(log_dic)
        return loss


class ThermalizerTrainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]
        self.softmax = nn.Softmax(dim=1)

        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=diffusion.Diffusion(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()

    def load_checkpoint(self,file_string,resume_wandb=True):
        """ Load checkpoint from saved file """
        with open(file_string, 'rb') as fp:
            model_dict = pickle.load(fp)
        assert model_dict["config"]==self.config, "Configs not the same"
        self.model=misc.load_diffusion_model(file_string).to(self.gpu_id)
        self._prep_optimizer()
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        self.epoch=model_dict["epoch"]
        self.training_step=model_dict["training_step"]

        if self.wandb_init==False and resume_wandb:
            self.resume_wandb()

        return

    def training_loop(self):
        """ Training loop for residual emulator """
        self.model.train()
        for j,image in enumerate(self.train_loader):
            image=image.to(self.gpu_id)
            #image=ds_train[batch_idx]
            #tot_samps+=image.shape[0]
            self.optimizer.zero_grad()
            noise=torch.randn_like(image).to(self.gpu_id)
            pred,_,t,pred_level=self.model(image,noise,True)
            loss_score=self.criterion(pred,noise)
            loss_classifier=F.cross_entropy(pred_level,t)
            loss=loss_score+self.lambda_c*loss_classifier
            loss.backward()

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1

            if self.ema:
                self.ema.update()

        return loss

    def valid_loop(self):
        raise NotImplementedError("Do not have a validation loop for thermalizer")

    def test_classifier(self):
        if self.ema:
            self.ema.apply_shadow()
        predicted_distribution=torch.zeros((self.config["timesteps"],self.config["timesteps"]))
        valid_image=self.valid_loader.dataset[:self.config["valid_samps"]].to(self.gpu_id)

        ## Classifier predictions - only do this once at the end as its slow
        with torch.no_grad():
            for aa in range(self.config["timesteps"]):
                noise=torch.randn_like(valid_image).to(self.gpu_id)
                noise_categories=(torch.ones(len(valid_image),device=self.gpu_id)*aa).to(torch.int64) ## True noise level
                noised_imgs=self.model._forward_diffusion(valid_image,noise_categories,noise)
                _,pred_noise_level=self.model.model(noised_imgs,True) ## Predicted noise levels
                predicted_distribution[aa]=self.softmax(pred_noise_level).mean(axis=0).cpu()

        dist_figure=plt.figure()
        plt.imshow(predicted_distribution)
        plt.colorbar()
        plt.xlabel("Predicted noise level")
        plt.ylabel("True noise level")
        wandb.log({"Classifier predictions":wandb.Image(dist_figure)})
        plt.close()

        if self.ema:
            self.ema.restore()

        return

    def save_checkpoint(self, checkpoint_string):
        """ Checkpoint model and optimizer """

        save_dict={
                    'epoch': self.epoch,
                    'training_step': self.training_step,
                    'state_dict': self.model.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config':self.config,
                    }
        with open(checkpoint_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def test_samples(self):
        ####### Classifier test
        self.model.eval()

        if self.ema:
            self.ema.apply_shadow()
            
        samples=self.model.sampling(40)

        if samples.isnan().any():
            print("Samples have nans, ignoring plot routines")
            return

        if self.config["PDE"]=="Kolmogorov":
            samples_fig=plt.figure(figsize=(18, 9))
            plt.suptitle("Samples after %d epochs" % self.epoch)
            for i in range(40):
                plt.subplot(5, 8, 1 + i)
                plt.axis('off')
                plt.imshow(samples[i].squeeze(0).data.cpu().numpy(),
                        cmap=sns.cm.icefire)
                plt.colorbar()
            plt.tight_layout()
            wandb.log({"Samples":wandb.Image(samples_fig)})
            plt.close()
            
            hist_figure=plt.figure()
            plt.suptitle("Sampled distribution after %d epochs" % self.epoch)
            plt.hist(samples.flatten().cpu(),bins=1000);
            wandb.log({"Hist":wandb.Image(hist_figure)})
            plt.close()
        
        else:
            samples_fig_u=plt.figure(figsize=(18, 9))
            plt.suptitle("Samples after %d epochs, upper layer" % self.epoch)
            for i in range(40):
                plt.subplot(5, 8, 1 + i)
                plt.axis('off')
                plt.imshow(samples[i][0].squeeze(0).data.cpu().numpy(),
                        cmap=sns.cm.icefire)
                plt.colorbar()
            plt.tight_layout()
            wandb.log({"Samples Upper":wandb.Image(samples_fig_u)})
            plt.close()
            
            samples_fig_l=plt.figure(figsize=(18, 9))
            plt.suptitle("Samples after %d epochs, upper layer" % self.epoch)
            for i in range(40):
                plt.subplot(5, 8, 1 + i)
                plt.axis('off')
                plt.imshow(samples[i][1].squeeze(0).data.cpu().numpy(),
                        cmap=sns.cm.icefire)
                plt.colorbar()
            plt.tight_layout()
            wandb.log({"Samples Lower":wandb.Image(samples_fig_l)})
            plt.close()
            
            hist_figure=plt.figure()
            plt.suptitle("Sampled distribution after %d epochs" % self.epoch)
            plt.hist(samples[:,0].flatten().cpu(),bins=1000,alpha=0.4,label="Upper layer");
            plt.hist(samples[:,1].flatten().cpu(),bins=1000,alpha=0.4,label="Lower layer");
            plt.legend()
            wandb.log({"Hist":wandb.Image(hist_figure)})
            plt.close()

        if self.ema:
            self.ema.restore()

    def run(self,epochs=None):
        if self.logging and self.wandb_init==False:
            self.init_wandb()
            self.model.model.config=self.config ## Update Unet config too, missed in parent call

        if epochs:
            max_epochs=epochs
        else:
            max_epochs=self.config["optimization"]["epochs"]

        for epoch in range(self.epoch,max_epochs+1):
            self.epoch=epoch
            print("Training at epoch", self.epoch)
            self.training_loop()
            self.test_samples()
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_last.p")
            if self.ema:
                self.ema.apply_shadow()
                self.save_checkpoint(self.config["save_path"]+"/checkpoint_last_ema.p")
                self.ema.restore()

        self.test_classifier()
            
        print("DONE on rank", self.gpu_id)
        return


class RefinerTrainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.residual=self.config["residual"]
        ## Make sure our model has timesteps if we are doing refiner!
        assert self.model.time_embedding, "No time embedding in Unet"
        self._build_refiner()
        self.test_criterion=torch.nn.MSELoss(reduction="none")

        if self.config["ddp"]==True:
            raise NotImplementedError
        if self.config["PDE"]=="Kolmogorov":
            ## Set up test trajectory tensor
            with open("/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/test40.p", 'rb') as fp:
                test_suite = pickle.load(fp)
            self.test_trajectories=(test_suite["data"][:,:200]/self.model.config["field_std"]).to(self.gpu_id)
        elif self.config["PDE"]=="QG":
            self.test_trajectories=torch.load("/scratch/cp3759/thermalizer_data/qg/test_eddy/eddy_dt5_20.pt")[:,:200]

    def _build_refiner(self):
        """ Set up pderefiner module """

        ## Some clunk imported from pdearena i don't have time to trim just yet
        from dataclasses import dataclass
        @dataclass
        class PDEDataConfig:
            n_scalar_components: int
            n_vector_components: int
            trajlen: int
            n_spatial_dim: int
            
        pdeconfig=PDEDataConfig(n_scalar_components=1,n_vector_components=0,trajlen=2,n_spatial_dim=2)
        self.refiner_model=refiner.PDERefiner(time_history=1,time_future=self.config["output_channels"],time_gap=0,max_num_steps=10,
                    criterion="mse",pdeconfig=pdeconfig,model=self.model,predict_difference=self.residual)
        return

    def training_loop(self):
        for batch in self.train_loader:
            xx=batch[:,0].to(self.gpu_id)
            yy=batch[:,1].to(self.gpu_id)
            if self.config["PDE"]=="Kolmogorov":
                xx=xx.unsqueeze(1)
                yy=yy.unsqueeze(1)
            if self.residual:
                yy=yy-xx
            self.optimizer.zero_grad()
            loss,_,_=self.refiner_model.train_step((xx,yy))
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["train_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return

    def test_rollout_kol(self,numsteps=None,wandb_figures=True):
        """ Run a short rollout alongside a test trajectory """

        if self.ema:
            self.ema.apply_shadow()
        xx=self.test_trajectories[:,0].unsqueeze(1)
        losses=[]
        
        if numsteps:
            trajlen=numsteps
        else:
            trajlen=len(self.test_trajectories[0])-1

        losses=torch.zeros((self.test_trajectories.shape[0],trajlen))

        ## Run short rollout
        with torch.no_grad():
            for aa in range(1,trajlen):
                xx=self.refiner_model.predict_next_solution(xx)
                losses[:,aa]=torch.mean(self.test_criterion(xx.squeeze(),self.test_trajectories[:,aa].to("cuda")),dim=(1,2))

        fig_samps=plt.figure(figsize=(20,6))
        samps=6
        plt.suptitle("Sim (top) and emu (bottom) after %d steps, at epoch %d" % (trajlen,self.epoch))
        for aa in range(1,samps):
            plt.subplot(2,samps,aa+1)
            plt.imshow(self.test_trajectories[aa,trajlen].squeeze().cpu(),cmap=sns.cm.icefire,interpolation='none')
            plt.colorbar()
        
            plt.subplot(2,samps,aa+1+samps)
            plt.imshow(xx[aa].squeeze().cpu(),cmap=sns.cm.icefire,interpolation='none')
            plt.colorbar()
        plt.tight_layout()
        if wandb_figures:
            wandb.log({"Random samples": wandb.Image(fig_samps)})
            plt.close()

        fig_losses=plt.figure()
        plt.title("Losses along short trajectory at epoch %d" % self.epoch)
        for aa in range(len(losses)):
            plt.semilogy(losses[aa],color="gray",alpha=0.5)
        plt.ylim(1e-4,1e3)
        if wandb_figures:
            wandb.log({"Loss samples": wandb.Image(fig_losses)})
            plt.close()

        if self.ema:
            self.ema.restore()
        return

    def test_rollout_qg(self,numsteps=None,wandb_figures=True):
        xx=self.test_trajectories[:,0].to("cuda")
        losses=[]

        if numsteps:
            trajlen=numsteps
        else:
            trajlen=len(self.test_trajectories[0])-1

        losses=torch.zeros((self.test_trajectories.shape[0],trajlen))

        ## Run short rollout
        with torch.no_grad():
            for aa in range(1,trajlen):
                xx=self.refiner_model.predict_next_solution(xx)
                losses[:,aa]=torch.mean(self.test_criterion(xx.squeeze(),self.test_trajectories[:,aa].to("cuda")),dim=(1,2,3))

        fig_samps=plt.figure(figsize=(20,6))
        samps=6
        plt.suptitle("Sim (top) and emu (bottom) after %d steps, at epoch %d" % (trajlen,self.epoch))
        for aa in range(1,samps):
            plt.subplot(2,samps,aa+1)
            plt.imshow(self.test_trajectories[aa,trajlen,0].squeeze().cpu(),cmap=sns.cm.icefire,interpolation='none')
            plt.colorbar()

            plt.subplot(2,samps,aa+1+samps)
            plt.imshow(xx[aa,0].squeeze().cpu(),cmap=sns.cm.icefire,interpolation='none')
            plt.colorbar()
        plt.tight_layout()
        if wandb_figures:
            wandb.log({"Random samples": wandb.Image(fig_samps)})
            plt.close()

        fig_losses=plt.figure()
        plt.title("Losses along short trajectory at epoch %d" % self.epoch)
        for aa in range(len(losses)):
            plt.semilogy(losses[aa],color="gray",alpha=0.5)
        plt.ylim(1e-4,1e3)
        if wandb_figures:
            wandb.log({"Loss samples": wandb.Image(fig_losses)})
            plt.close()
        return

    def save_checkpoint(self, checkpoint_string):
        """ Checkpoint model and optimizer """

        save_dict={
                    'epoch': self.epoch,
                    'training_step': self.training_step,
                    'state_dict': self.refiner_model.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config':self.config,
                    }
        with open(checkpoint_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def run(self,epochs=None):
        """ Can run for a custom number of epochs, otherwise will run for however many is in config """
        if self.logging and self.wandb_init==False:
            self.init_wandb()
        
        if epochs:
            max_epochs=epochs
        else:
            max_epochs=self.config["optimization"]["epochs"]

        for epoch in range(self.epoch,max_epochs+1):
            self.epoch=epoch
            self.training_loop()
            if self.logging:
                if self.config["PDE"]=="Kolmogorov":
                    self.test_rollout_kol()
                else:
                    self.test_rollout_qg()
                self.save_checkpoint(self.config["save_path"]+"/checkpoint_last.p")
                if self.ema:
                    self.ema.apply_shadow()
                    self.save_checkpoint(self.config["save_path"]+"/checkpoint_last_ema.p")
                    self.ema.restore()
        print("DONE on rank", self.gpu_id)
        return
