import torch
import json
import os
import numpy as np
from PIL import Image
from modules import VAE, DepthDataset, RGBDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('..')

if __name__ == "__main__":

    log_dir = "VAE/tmp_eval/repeat_best"
    mode = "training"
    os.makedirs(log_dir, exist_ok=True) 

    # hyperparameters
    in_channels = 3
    latent_dim = 16
    layers = [32, 64, 128, 256, 512]
    lr = 1e-3
    batch_size = 16
    kld_weight = 0.00025
    activation = "LeakyReLU"
    batch_norm = "BatchNorm2d"
    output_activation = "Tanh"
    tf = ["RandomVerticalFlip", "RandomHorizontalFlip"]

    # create model
    model = VAE(in_channels=in_channels, latent_dim=latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    img_path = glob.glob("images/images_rgb/*.png")

    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(90),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = RGBDataset(img_path, transform)
    train_len = 448
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # TRAIN MODEL

    if mode == "training":

        num_epochs = 300
        lowest_loss = 1e10
        loss_list = []
        eval_loss_list = []

        for epoch in range(num_epochs):

            # train the model

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

            # evaluate the model

            model.eval()
            eval_epoch_loss = 0

            for batch_images in val_loader:
                batch_images = batch_images.float().to(device)
                recon_images, mu, logvar = model(batch_images)
                loss = model.loss_function(recon_images, batch_images, mu, logvar)
                eval_epoch_loss += loss.item()
            eval_epoch_loss /= len(val_loader)
            eval_loss_list.append(eval_epoch_loss)

            print(f"Epoch {epoch+1} train loss: {epoch_loss:.4f}, eval loss: {eval_epoch_loss:.4f}")
            # if best model so far, save it
            if eval_epoch_loss < lowest_loss:
                lowest_loss = eval_epoch_loss
                torch.save(model.state_dict(), log_dir + "/best_model.pt")
                print(f"New best model saved with loss: {lowest_loss:.4f}")

        # plot the loss
        plt.plot(loss_list, label="train loss")
        plt.plot(eval_loss_list, label="eval loss")
        plt.title("Loss por época de entrenamiento")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.show()

        config = dict(
            in_channels=in_channels,
            latent_dim=latent_dim,
            layers=layers,
            activation=activation,
            batch_norm=batch_norm,
            output_activation=output_activation,
            lr = lr,
            kld_weight = kld_weight,
            batch_size = batch_size,
            transforms = tf,
            lowest_loss=lowest_loss
        )
        json_object = json.dumps(config, indent=4)
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            f.write(json_object)

        np.save(os.path.join(log_dir, 'train_loss.npy'), loss_list)
        np.save(os.path.join(log_dir, 'eval_loss.npy'), eval_loss_list)

    # TEST MODEL

    # load the model
    model.load_state_dict(torch.load(log_dir + "/best_model.pt"))
    for batch_images in val_loader:
        for n in range(len(batch_images)):
            test_img = batch_images[n]
            #test_img = (test_img - test_img.min()) / (test_img.max() - test_img.min())
            input = test_img.float().unsqueeze(0).to(device)
            mu, logvar = model.encode(input)
            recon_img = model.decode(mu)
            recon_img = recon_img.squeeze().detach()
            # plot input and output
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(transforms.ToPILImage()(test_img))
            axes[1].imshow(transforms.ToPILImage()(recon_img))
            plt.show()