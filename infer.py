import os
import argparse
import numpy as np
import torch
import torchvision
from glob import glob
from model import create_model
from PIL import Image
from transforms import get_val_compose
from torch.utils.data import DataLoader
from pl_module import ClassifierNet
import pytorch_lightning as pl
import utils
import datasets

DEFAULT_GPUS = min(1, torch.cuda.device_count())
DEFAULT_NUM_WORKERS = int(os.cpu_count() / 2)
DEFAULT_BACKBONE_NAME = "resnet34"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TILE_SIZE = 256
CHECKPOINTS_ROOT = "./trained_models"

id_to_class = datasets.id_to_class()


def get_trained_model(backbone_name, checkpoint):
    model = ClassifierNet(model_name=backbone_name, num_classes=3)
    model = model.load_from_checkpoint(checkpoint)
    model.eval()
    return model

def infer_single_image(model, image_location, transform_val):
    if not os.path.isfile(image_location):
        print(f"ERROR: Can't read the image location = '{image_location}'")
        exit(0)

    # 1. Open
    img = Image.open(image_location).convert('RGB')
    # 2. Transform to tensor, etc
    img = transform_val(img).unsqueeze(0)

    if DEFAULT_GPUS == 0:
        img = img.cpu()
    else:
        img = img.cuda()

    # 3. Run the NN and read the output
    out = model(img).data.cpu().numpy()[0]
    index = np.argmax(out)
    predict_class = id_to_class[index]

    print(f"The image at path:  '{image_location}'")
    print(f"Was predicted as:   {index}:'{predict_class}'")
    print(f"Original output is: {out}")
    print(" = " * 40)
    return f"{index}:{predict_class}"


def run_prediction(
        backbone_name,
        checkpoint,
        image_location=None,
        directory_location=None,
):
    model = get_trained_model(backbone_name, checkpoint)
    transform_val = get_val_compose(DEFAULT_TILE_SIZE)

    results = {}
    if image_location:
        res_class = infer_single_image(model, image_location, transform_val)
        results[image_location] = res_class

    if directory_location:
        if directory_location[-1] != "/":
            directory_location += "/"
        images = sorted(glob(directory_location + "*png"))
        for img in images:
            res_class = infer_single_image(model, img, transform_val)
            results[img] = res_class

    return results

def make_parser():
    """Make and return the argument parser"""
    parser = argparse.ArgumentParser(
        description="Run inference for a single image, or for all the images\n"
                    "inside a directory. It can use one of the trained included models\n"
                    "or use a checkpoint trained by the user.\n"
                    "Either --image_location or --directory_location must be provided.\n"
                    "By default, results are stored at 'last_results.csv', but this path\n"
                    "can be selected with --result_csv_file.\n"
    )
    parser.add_argument("-b", "--backbone_name",
                        help="The name of the backbone for the classifier. One of the following: "
                             "['resnet18'|'resnet34'|'resnet50'|'vit_b_16'|'convnext_tiny'"
                             "|'efficientnet_b3'|'efficientnet_b5']. Default and recommended is "
                             "'resnet34'.",
                        default=DEFAULT_BACKBONE_NAME)
    parser.add_argument("-c", "--checkpoint_path",
                        help="Location of the checkpoint to use. If not provided, it will"
                             "use the best weights included for the particular backbone",
                        type=str)
    parser.add_argument("-i", "--image_location",
                        help="Location of a single image to run the prediction",
                        type=str)
    parser.add_argument("-d", "--directory_location",
                        help="Location of a directory to run inference in all the PNG images.",
                        type=str)
    parser.add_argument("-w", "--workers",
                        help="The number of workers for the DataLoaders. Default is CPU_COUNT/2.",
                        type=int,
                        default=DEFAULT_NUM_WORKERS)
    parser.add_argument("-g", "--gpus",
                        help="The number of GPUS to use. By default it will use what PyTorch finds",
                        type=int,
                        default=DEFAULT_GPUS)
    parser.add_argument("-s", "--batch_size",
                        help="Batch size",
                        type=int,
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-r", "--result_csv_file",
                        help="Path of a csv file to save the results",
                        type=str,
                        default="last_results.csv")
    return parser


if __name__ == "__main__":
    argument_parser = make_parser()
    args = argument_parser.parse_args()

    backbone_name = args.backbone_name
    checkpoint_path = args.checkpoint_path
    image_location = args.image_location
    directory_location = args.directory_location
    workers = args.workers
    gpus = args.gpus
    batch_size = args.batch_size
    result_csv_file = args.result_csv_file

    if not image_location and not directory_location:
        print("ERROR: You must provide -i/--image_location or -d/--directory_location")
        exit(0)

    if not checkpoint_path:
        best_checkpoint_filename = utils.best_checkpoint_per_backbone(backbone_name)
        if not best_checkpoint_filename:
            exit(0)

        checkpoint_path = os.path.join(
            f"{CHECKPOINTS_ROOT}/{backbone_name}/",
            best_checkpoint_filename
        )
        print("WARNING: Checkpoint path not provided, going to use the pretrained")

    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Can't read the checkpoint_path = '{checkpoint_path}'")
        exit(0)

    print("MESSAGE: Checkpoint used:")
    print(f"        {checkpoint_path}")

    results = run_prediction(
        backbone_name,
        checkpoint_path,
        image_location=image_location,
        directory_location=directory_location,
    )

    utils.write_results(results, result_csv_file)
