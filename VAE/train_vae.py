import torch
from modules import VAE, DepthDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('..')

if __name__ == "__main__":

    # load the vae model / gpu if available
    model = VAE()
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
    train_set, val_set = torch.utils.data.random_split(dataset, [256, 16])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

    '''
    # load the model
    model.load_state_dict(torch.load("models/model.pt"))
    for batch_images in val_loader:
        test_img = batch_images[0]
        test_img = (test_img - test_img.min()) / (test_img.max() - test_img.min())
        input = test_img.float().unsqueeze(0).to(device)
        recon_img, _, _ = model(input)
        recon_img = recon_img.squeeze().cpu().detach().numpy()
        # plot input and output
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test_img.squeeze(), cmap="gray")
        axes[1].imshow(recon_img, cmap="gray")
        plt.show()
    '''

    '''
    for batch_images in train_loader:
        for n in range(len(batch_images)):
            plt.imshow(batch_images[n].squeeze().numpy())
            plt.show()
    '''

    # train the model
    num_epochs = 300
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_images in train_loader:
            batch_images = batch_images.float().to(device)
            batch_images = (batch_images - batch_images.min()) / (batch_images.max() - batch_images.min())
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
    torch.save(model.state_dict(), "models/model.pt")


