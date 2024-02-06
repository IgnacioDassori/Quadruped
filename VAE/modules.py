import torch
import os
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=16, lr=5e-3, kld_weight=0.00025):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.kld_weight = kld_weight
        if in_channels == 1:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
            )
            self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
            self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
            self.fc_decoder = nn.Linear(latent_dim, 256 * 4 * 4)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
                nn.Sigmoid(),
            )
        elif in_channels == 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
            )
            self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
            self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
            self.fc_decoder = nn.Linear(latent_dim, 512 * 4 * 4)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
                nn.Tanh(),
            )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 512, 4, 4)
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        REL = F.mse_loss(recon_x, x, reduction="sum")
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return REL + KLD*self.kld_weight
    

class VAE_512(nn.Module):
    def __init__(self, in_channels=3, latent_dim=16, lr=5e-3, kld_weight=0.00025, hidden_dims=[32, 64, 128, 256, 512]):
        super(VAE_512, self).__init__()

        self.latent_dim = latent_dim
        self.lr = lr
        self.kld_weight = kld_weight


        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 4, stride = 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        reduced_size = int(512 / (2 ** len(hidden_dims)))
        self.reduced_size = reduced_size
        self.fc_mu = nn.Linear(hidden_dims[-1] * reduced_size * reduced_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * reduced_size * reduced_size, latent_dim)

        # Build Decoder
        modules = []
        self.fc_decoder = nn.Linear(latent_dim, hidden_dims[-1] * reduced_size * reduced_size)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride = 2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels= in_channels,
                                   kernel_size= 4, stride= 2, padding= 1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  
        self.hidden_dims = hidden_dims
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, self.hidden_dims[0], self.reduced_size, self.reduced_size)
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        REL = F.mse_loss(recon_x, x, reduction="sum")
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return REL + KLD*self.kld_weight


class DeepAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super(DeepAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.latent_fc = nn.Linear(256 * 4 * 4, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.latent_fc(x)
        x = self.decoder_fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x
    
class AutoencoderFC(nn.Module):
    def __init__(self, latent_dim=16):
        super(AutoencoderFC, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64 * 64),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x
    
class DepthDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.image_path[idx])
        if self.transform:
            tensor = self.transform(tensor)
        return tensor
    
class RGBDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        image = Image.open(img_name).resize((128, 128)).convert('RGB')
        if self.transform:
            tensor = self.transform(image)
        return tensor
    
class HouseDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            tensor = self.transform(image)
        return tensor