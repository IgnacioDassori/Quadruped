import torch
torch.cuda.empty_cache()
import json
import os
from modules import VAE, VAE_512, RGBDataset, HouseDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('..')

if __name__ == "__main__":

    log_dir = "VAE/results_house/dropout11"
    mode = "eval"
    os.makedirs(log_dir, exist_ok=True) 

    # hyperparameters
    in_channels = 3
    latent_dim = 32
    layers = [16, 32, 64, 128, 256]
    lr = 1e-4
    kld_weight = 0.00025
    batch_size = 4
    activation = "LeakyReLU"
    batch_norm = "BatchNorm2d"
    output_activation = "Tanh"
    tf = ["RandomVerticalFlip", "RandomHorizontalFlip"]
    grayscale = False
    if grayscale:
        in_channels = 1

    # create model
    model = VAE_512(in_channels=in_channels, 
                    latent_dim=latent_dim, 
                    lr=lr,
                    kld_weight=kld_weight,
                    hidden_dims=layers,
                    output_activation=output_activation)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    img_path = glob.glob("images/house_rgb_512/*.jpg")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]) if not grayscale else transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset = HouseDataset(img_path, transform)
    train_len = 2048
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
        since_last_best = 0

        for epoch in range(num_epochs):

            # train the model

            model.train()
            epoch_loss = 0
            for batch_images in train_loader:
                batch_images = batch_images.float().to(device)
                recon_images, mu, logvar = model(batch_images)
                loss = model.loss_function(recon_images, batch_images, mu, logvar)
                epoch_loss += loss.item()
                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
            since_last_best += 1
            if eval_epoch_loss < lowest_loss:
                since_last_best = 0
                lowest_loss = eval_epoch_loss
                torch.save(model.state_dict(), log_dir + "/best_model.pt")
                print(f"New best model saved with loss: {lowest_loss:.4f}")

            if since_last_best > 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

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
            layers=layers.reverse(),
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



    # TEST MODEL
    max = 0
    min = 0
    # load the model
    model.load_state_dict(torch.load(log_dir + "/best_model.pt"))
    for batch_images in val_loader:
        for n in range(len(batch_images)):
            test_img = batch_images[n]
            input = test_img.float().unsqueeze(0).to(device)
            recon_img, _, _ = model(input)
            fig, axes = plt.subplots(1, 2)
            recon_img = recon_img.squeeze()
            axes[0].imshow(transforms.ToPILImage()(test_img))
            axes[1].imshow(transforms.ToPILImage()(recon_img))
            plt.show()      