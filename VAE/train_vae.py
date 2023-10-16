import torch
import json
import os
from PIL import Image
from modules import VAE, DepthDataset, RGBDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('..')

if __name__ == "__main__":

    log_dir = "tmp/first_vae"
    os.makedirs(log_dir, exist_ok=True) 

    # load the vae model / gpu if available
    in_channels = 3
    latent_dim = 16
    model = VAE(in_channels=in_channels, latent_dim=latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    img_path = glob.glob("images/images_rgb/*.png")

    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = RGBDataset(img_path, transform)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)
    
    # load the model
    model.load_state_dict(torch.load(log_dir + "/model.pt"))
    for batch_images in val_loader:
        test_img = batch_images[0]
        #test_img = (test_img - test_img.min()) / (test_img.max() - test_img.min())
        input = test_img.float().unsqueeze(0).to(device)
        recon_img, _, _ = model(input)
        recon_img = recon_img.squeeze().cpu().detach().numpy()
        # plot input and output
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(transforms.ToPILImage(test_img), cmap="gray")
        axes[1].imshow(transforms.ToPILImage(recon_img), cmap="gray")
        plt.show()


# TRAIN MODEL
'''
    # train the model
    log_dir = "tmp/first_vae"
    os.makedirs(log_dir, exist_ok=True) 
    num_epochs = 300
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_images in train_loader:
            batch_images = batch_images.float().to(device)
            #batch_images = (batch_images - batch_images.min()) / (batch_images.max() - batch_images.min())
            recon_images, mu, logvar = model(batch_images)
            loss = model.loss_function(recon_images, batch_images, mu, logvar)
            epoch_loss += loss.item()
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
        epoch_loss /= len(train_loader)
        loss_list.append(epoch_loss)

        print(f"Epoch {epoch+1} train loss: {epoch_loss:.4f}")

    # plot the loss
    plt.plot(loss_list)
    plt.title("Loss por época de entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.show()
    # save the model
    torch.save(model.state_dict(), log_dir + "/model.pt")

'''
