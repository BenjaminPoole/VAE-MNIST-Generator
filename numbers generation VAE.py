import numpy as np
import matplotlib.pyplot as plt

from autoencoder import VAE
from train import load_mnist
import pyautogui
import cv2

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    autoencoder = VAE.load("VAE_model")
    #x_train, y_train, x_test, y_test = load_mnist()

    while True:
        (x_coord, y_coord)=pyautogui.position()
        x_coord = (x_coord - 960)*0.004
        y_coord = (y_coord - 540)*0.006
        
        coords = np.asarray([[x_coord,y_coord]])
        
        image = autoencoder.generate(coords)
        image = image.squeeze()
        image = cv2.resize(image,(140,140),interpolation=None)
        # print(image)
        cv2.imshow('VAE Image', image)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


    # num_sample_images_to_show = 8
    # sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    # print(sample_images.shape)
    # reconstructed_images, space = autoencoder.reconstruct(sample_images)
    # print(space)
    # plot_reconstructed_images(sample_images, reconstructed_images)

    # _, latent_representations = autoencoder.reconstruct(x_test)
    # plot_images_encoded_in_latent_space(latent_representations, y_test)




















