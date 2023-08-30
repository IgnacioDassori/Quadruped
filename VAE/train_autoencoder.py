import torch
from torch import nn
from modules import DeepAutoencoder, DepthDataset
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

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