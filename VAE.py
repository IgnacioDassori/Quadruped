import torch
from torch import nn
from torch.nn import functional as F
from dataset import DepthDataset
import matplotlib.pyplot as plt
import glob

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=8):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.kld_weight = 0.01
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 256, 4, 4)
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
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return REL + KLD
    
if __name__ == "__main__":

    # load the vae model / gpu if available
    model = VAE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    img_path = glob.glob("images/images_pt/*.pt")
    dataset = DepthDataset(img_path)
    train_set, val_set = torch.utils.data.random_split(dataset, [250, 28])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

    # train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_images in train_loader:
            batch_images = (batch_images.float()/255.0).unsqueeze(1).to(device)
            recon_images, mu, logvar = model(batch_images)
            loss = model.loss_function(recon_images, batch_images, mu, logvar)
            epoch_loss += loss.item()
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
        epoch_loss /= len(train_loader)

        # evaluate the model
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch_images in val_loader:
                batch_images = (batch_images.float()/255.0).unsqueeze(1).to(device)
                recon_images, mu, logvar = model(batch_images)
                loss = model.loss_function(recon_images, batch_images, mu, logvar)
                eval_loss += loss.item()
        eval_loss /= len(val_loader)

        print(f"Epoch {epoch+1} train loss: {epoch_loss:.4f} eval loss: {eval_loss:.4f}")

    for batch_images in val_loader:
        test_img = batch_images[0]
        input = (test_img.float()/255.0).unsqueeze(0).unsqueeze(0).to(device)
        recon_img, _, _ = model(input)
        recon_img = recon_img.squeeze().cpu().detach().numpy()*255.0
        # plot input and output
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test_img, cmap="gray")
        axes[1].imshow(recon_img, cmap="gray")
        plt.show()