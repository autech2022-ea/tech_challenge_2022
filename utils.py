import os
import gdown
import numpy as np
import matplotlib
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

DEFAULT_CHECKPOINT_LOCATION = './trained_models'

def download_checkpoint(url, target_dir, target_name):
    """Use gdown to download a file given a fuzzy share URL"""
    output = os.path.join(target_dir, target_name)
    gdown.download(url, output, quiet=False, fuzzy=True)


def best_checkpoint_per_backbone(backbone_name):
    """Return the predefined best model per backbone. Downloads the file is not present"""
    backbone_root = f"{DEFAULT_CHECKPOINT_LOCATION}/{backbone_name}"

    if backbone_name == "resnet34":
        name = "val_acc=0.974-val_loss=0.799-epoch=31.ckpt"
        gdrive_url = "https://drive.google.com/file/d/1JfMkHFBjV2cTGHMM65JSnJN8fnFHYW2i/view?usp=sharing"
    elif backbone_name == "resnet18":
        name = "val_acc=0.910-val_loss=0.358-epoch=35.ckpt"
        gdrive_url = "https://drive.google.com/file/d/1cuP9Y5altaNoTS-y3bVBIkU2Sy8MILka/view?usp=sharing"
    elif backbone_name == "convnext_tiny":
        name = "val_acc=0.923-val_loss=0.237-epoch=24.ckpt"
        gdrive_url = "https://drive.google.com/file/d/1NQIZvGYXbrFPkOhUDZK8Br8dPFxOksQd/view?usp=sharing"
    else:
        print("ERROR: Sorry, we don't have trained models for that backbone yet")
        return None

    if not os.path.isfile(os.path.join(backbone_root, name)):
        print("WARNING: File not present, downloading from Google Drive")
        download_checkpoint(
            gdrive_url, backbone_root, name
        )
    return name


def show_images(
    images,
    figsize=(10, 10),
    columns=1,
    title=None,
    save_path=None
):
    """
    Plot a series of images as matplotlib plot. Each as a subplot. The
    images can be given as np arrays (if open by OpenCV), or as the path
    to open them, given as a str. Optionally the plot can be saved as
    a png file if save_path is given.

    :param list images: list of images, each image can be np array, or the path as str
    :param tuple figsize: a tuple of two numbers to set the dimensions of the plot
    :param int columns: number of colums (how many plots per line)
    :param str title: the global title of the plot
    :param str save_path: path to save the image
    :return: None
    """
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=20)
    for i, image in enumerate(images):
        subplot = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        if isinstance(image, np.ndarray):
            plt.imshow(image)
        elif isinstance(image, str):
            img = mpimg.imread(image)
            plt.imshow(img)
    if save_path:
        plt.savefig(save_path)

def write_results(results, result_csv_file):
    with open(result_csv_file, "w") as file:
        for k, v in results.items():
            file.write(f"{k},{v}\n")
