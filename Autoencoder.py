import torch
from torch import nn
from torch.nn import functional as F
from dataset import DepthDataset
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

class DeepAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super(DeepAutoencoder, self).__init__()
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
        self.latent_fc = nn.Linear(256 * 4 * 4, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
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
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 64),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x
    
if __name__ == "__main__":

    model = DeepAutoencoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    img_path = glob.glob("images/images_pt/*.pt")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    dataset = DepthDataset(img_path, transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [240, 32])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

    # train the model
    criterion = nn.MSELoss()
    num_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_images in train_loader:
            batch_images = batch_images.float().to(device)
            recon_images = model(batch_images)
            loss = criterion(recon_images, batch_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1} train loss: {epoch_loss:.4f}")

    for batch_images in val_loader:
        test_img = batch_images[0]
        input = test_img.float().unsqueeze(0).to(device)
        recon_img = model(input)
        recon_img = recon_img.squeeze().cpu().detach().numpy()
        # plot input and output
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test_img.squeeze(), cmap="gray")
        axes[1].imshow(recon_img, cmap="gray")
        plt.show()