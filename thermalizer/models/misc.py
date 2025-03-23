import numpy as np
import io
import pickle
import xarray as xr
import math
import torch
from torch import nn
from tqdm import tqdm

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


import thermalizer.models.diffusion as diffusion
import thermalizer.models.cnn as cnn
import thermalizer.models.unet_modern as munet
import thermalizer.models.drn as drn

### Model factory
def model_factory(config):
    """ Function to take a config dict, and return one of our nn.Modules """
    if config["model_type"]=="ModernUnet":
        return munet.ModernUnet(config)
    elif config["model_type"]=="ModernUnetRegressor":
        return munet.ModernUnetRegressor(config)
    elif config["model_type"]=="DRN":

        return drn.ResNet(config)
    else:
        print("Model type not recognised")
        quit()

## Activation registry for resnet modules
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


## Noise timestep embedding
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    Retrieved from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#LL90C1-L109C13
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    assert emb.shape == (timesteps.shape[0], embedding_dim), f"{emb.shape}"
    return emb


## Mnist helper functions
def create_mnist_datasets(image_size=32,single_number=None):
    """ Return mnist datasets (not dataloaders)
        image_size:   resolution of images
        single_numer: if we want to return a dataset containing only
                      a single class of the mnist numbers

        returns: train_dataset, test_dataset
    """
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor()])

    train_dataset=MNIST(root="/scratch/cp3759/mnist/",\
                        train=True,\
                        download=False,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="/scratch/cp3759/mnist/",\
                        train=False,\
                        download=False,\
                        transform=preprocess
                        )

    if single_number is not None:
        single_digit_dexes=torch.where(train_dataset.targets==single_number)[0]
        train_dataset.data=train_dataset.data[single_digit_dexes]
        single_digit_dexes=torch.where(test_dataset.targets==single_number)[0]
        test_dataset.data=test_dataset.data[single_digit_dexes]
        
    return train_dataset, test_dataset

def create_mnist_dataloaders(batch_size,image_size=32,num_workers=8,single_number=None,test_batch_size=None):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor()])

    train_dataset=MNIST(root="/scratch/cp3759/mnist/",\
                        train=True,\
                        download=False,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="/scratch/cp3759/mnist/",\
                        train=False,\
                        download=False,\
                        transform=preprocess
                        )

    if single_number is not None:
        single_digit_dexes=torch.where(train_dataset.targets==single_number)[0]
        train_dataset.data=train_dataset.data[single_digit_dexes]
        single_digit_dexes=torch.where(test_dataset.targets==single_number)[0]
        test_dataset.data=test_dataset.data[single_digit_dexes]

    if test_batch_size:
        test_batch=test_batch_size
    else:
        test_batch=batch_size

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=test_batch,shuffle=True,num_workers=num_workers)

########################## Loading models ################################
## Torch models trained using cuda and then pickled cannot be loaded
## onto cpu using the normal pickle methods: https://github.com/pytorch/pytorch/issues/16797
## This method replaces the pickle.load(input_file), using the same syntax
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_model(file_string):
    """ Load a pickled model, either on gpu or cpu """
    with open(file_string, 'rb') as fp:
        if torch.cuda.is_available():
            model_dict = pickle.load(fp)
        else:
            model_dict = CPU_Unpickler(fp).load()

    ## Check which model type to construct
    try:
        if model_dict["config"]["model_type"]=="ModernUnet":
            model=munet.ModernUnet(model_dict["config"])
        elif model_dict["config"]["model_type"]=="ModernUnetRegressor":
            model=munet.ModernUnetRegressor(model_dict["config"])
        elif model_dict["config"]["model_type"]=="DRN":
            model=drn.ResNet(model_dict["config"])
        elif model_dict["config"]["model_type"]=="FNO":
            model=fno.FNO(model_dict["config"])
        else:
            model=cnn.FCNN(model_dict["config"])
    except:
        model=cnn.FCNN(model_dict["config"])

    ## Load state_dict
    model.load_state_dict(model_dict["state_dict"])
    return model


def load_diffusion_model(file_string):
    """ Load a diffusion model. Read config file from the pickle
        Reconstruct the CNN, then use same config file to create
        a diffusion model with the loaded CNN """

    with open(file_string, 'rb') as fp:
        if torch.cuda.is_available():
            model_dict = pickle.load(fp)
        else:
            model_dict = CPU_Unpickler(fp).load()
    if model_dict["config"]["model_type"]=="ModernUnet":
        model_cnn=munet.ModernUnet(model_dict["config"])
        model_cnn.load_state_dict(model_dict["state_dict"])
    elif model_dict["config"]["model_type"]=="ModernUnetRegressor":
        model_cnn=munet.ModernUnetRegressor(model_dict["config"])
        model_cnn.load_state_dict(model_dict["state_dict"])

    diffusion_model=diffusion.Diffusion(model_dict["config"], model=model_cnn)
    return diffusion_model


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, overwrite=False):
        if len(self.shadow) > 0 and not overwrite:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.detach() + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        if len(self.shadow) == 0:
            print("Warning: EMA shadow is empty. Cannot apply shadow.")
        else:
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.data
                    param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        

def estimate_covmat(field_tensor,nsamp=None):
    """ Estimate covariance matrix from some tensor of fields. Can either be
        flattened to batched 1D tensors, or batched 2D fields 
        use nsamp to estimate covmat from a subsample of the data """

    ## If nsamp is not provided, use every sample in field_tensor
    if nsamp==None:
        nsamp=len(field_tensor)

    ## If the field tensor isn't flattened, flatten
    if len(field_tensor.shape)>2:
        field_tensor=field_tensor.reshape((len(field_tensor),64*64))

    ## Initialise covariance matrix
    cov=torch.zeros((64**2,64**2))

    for aa in tqdm(range(nsamp)):
        cov+=torch.outer(field_tensor[aa],field_tensor[aa])
    cov/=(nsamp-1)
    return cov


def get_whitening_from_cov(cov):
    """ Get whitening matrix from covariance matrix
        We are using the fact that we can diagonalise
        covariance matrix K=ULU^T

        Which allows us to obtain:
        W=K^{-1/2}=UL^{-1/2}U^T

        and produce a whitened random vector z=Wx with
        identity covariance.

        Return both the whitening and dewhitening matrices
    """
    
    ## Get eigenvalue decomposition of covariance matrix
    lamb,u=torch.linalg.eig(cov)
    ## Find L^{-1/2}
    lambinv=torch.eye(len(lamb))*(1/torch.sqrt(lamb))
    lambroot=torch.eye(len(lamb))*(torch.sqrt(lamb))

    ## W=K^{-1/2}=UL^{-1/2}U^T
    whitener=torch.matmul(torch.matmul(u,lambinv),u.T)
    dewhitener=torch.matmul(torch.matmul(u,lambroot),u.T)

    return whitener, dewhitener
    

class FieldNoiser():
    """ Forward diffusion module for various different noise schedulers """
    def __init__(self,timesteps,scheduler):
        self.timesteps=timesteps
        self.scheduler=scheduler
        #print(self.timesteps)

        if self.scheduler=="cosine":
            self.betas=self._cosine_variance_schedule(self.timesteps)
        elif self.scheduler=="linear":
            self.betas=self._linear_variance_schedule(self.timesteps)
        elif self.scheduler=="sigmoid":
            self.betas=self._sigmoid_variance_schedule(self.timesteps)


        self.alphas=1.-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,dim=-1)
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod)

        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            self.betas=self.betas.to(self.device)
            self.alphas=self.alphas.to(self.device)
            self.alphas_cumprod=self.alphas_cumprod.to(self.device)
            self.sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(self.device)
            self.sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(self.device)

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

    def __linear_variance_schedule(self,timesteps):
        betas=torch.linspace(0,1,steps=timesteps+1,dtype=torch.float32)
        return betas

    def _linear_variance_schedule(self,timesteps):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
        
    def _sigmoid_variance_schedule(self,timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def forward_diffusion(self,x_0,t,noise):
        """ Add noise to a clean field for t noise timesteps """
        assert x_0.shape==noise.shape, "Noise and fields have different shapes"
        #q(x_{t}|x_{0})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
