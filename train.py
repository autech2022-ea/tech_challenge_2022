import os
import argparse
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pl_module import ClassifierNet
import pytorch_lightning as pl
import datasets

DEFAULT_GPUS = min(1, torch.cuda.device_count())
DEFAULT_NUM_WORKERS = int(os.cpu_count() / 2)
DEFAULT_BACKBONE_NAME = "resnet34"
DEFAULT_CHECKPOINTS_LOCATION = "./checkpoints"
DEFAULT_DATASET_LOCATION = "./data"
TEST_SET_DIR = "./test_dataset/"
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 16
DEFAULT_TILE_SIZE = 256


def get_callbacks(backbone, checkpoints_location):
    """
    Get the list of callbacks.
    - Monitor the LR each epoch: LearningRateMonitor
    - Early stopping if val_acc is not decreasing: EarlyStopping
    - Save the last K models: checkpoint_callback
    """
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping('val_acc',
                                            patience=25,
                                            check_on_train_epoch_end=True,
                                            mode='max')
    checkpoint_callback = ModelCheckpoint(
        filename='{val_acc:.3f}-{val_loss:.3f}-{epoch}',
        dirpath=f"./{checkpoints_location}/{backbone}/",
        save_top_k=3, mode="max",
        monitor="val_acc"
    )
    return [lr_monitor_callback, early_stopping_callback, checkpoint_callback]


def run_train(
    backbone_name,
    checkpoints_location,
    learning_rate,
    batch_size,
    max_epochs,
    gpus,
    workers,
    dataset_location
):
    callbacks = get_callbacks(backbone_name, checkpoints_location)

    model = ClassifierNet(lr=learning_rate,
                          model_name=backbone_name,
                          num_classes=3)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=3,
        log_every_n_steps=10,
        max_epochs=max_epochs,
        gpus=gpus,
        logger=TensorBoardLogger(
            "lightning_logs/",
            name=f"{backbone_name}",
            flush_secs=10
        ),
        callbacks=callbacks
    )

    # For transformers, we need the size of 224 instead of 256
    if 'vit' not in backbone_name:
        tile_size = DEFAULT_TILE_SIZE
    else:
        tile_size = 224

    (train_set, train_sampler), (val_set, val_sampler) = datasets.get_datasets(
        dataset_location,
        val_ratio=0.15,
        tile_size=tile_size
    )

    trainer.fit(
        model,
        DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=workers,
            sampler=train_sampler
        ),
        DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=workers,
            sampler=val_sampler
        )
    )

    print("FINISHED TRAINING")
    checkpoint_callbacks = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
    if len(checkpoint_callbacks):
        print("This is the path of the best model (per val_acc):")
        print(f"{checkpoint_callbacks[0].best_model_path}")
    print(" = " * 50)

    print("Going to test with the TEST_SET:")
    (test_dataset, _), (_, _) = datasets.get_datasets(
        TEST_SET_DIR,
        tile_size=tile_size
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    test_set_metrics = trainer.test(model, test_dataloader, verbose=False)[0]
    print(f"TEST_SET Metrics = {test_set_metrics}")


def make_parser():
    """Make and return the argument parser"""
    parser = argparse.ArgumentParser(
        description="Run the training process for the model.\n"
                    "There are reasonable default for all the flags.\n"
                    "At the end of the training phase, the path of the "
                    "best trained model will be printed.\n"
    )
    parser.add_argument("-b", "--backbone_name",
                        help="The name of the backbone for the classifier. One of the following: "
                             "['resnet18'|'resnet34'|'resnet50'|'vit_b_16'|'convnext_tiny'"
                             "|'efficientnet_b3'|'efficientnet_b5']. Default and recommended is "
                             "'resnet34'.",
                        default=DEFAULT_BACKBONE_NAME)
    parser.add_argument("-e", "--max_epochs",
                        help="Maximum number of epochs. It may not reach the number because of "
                             "the EarlyStopping callback.",
                        type=int,
                        default=DEFAULT_EPOCHS)
    parser.add_argument("-g", "--gpus",
                        help="The number of GPUS to use. By default it will use what PyTorch finds",
                        type=int,
                        default=DEFAULT_GPUS)
    parser.add_argument("-l", "--learning_rate",
                        help=f"The learning rate. Default is {DEFAULT_LR}.",
                        type=float,
                        default=DEFAULT_LR)
    parser.add_argument("-s", "--batch_size",
                        help="Batch size",
                        type=int,
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-c", "--checkpoints_location",
                        help="Location to store the checkpoints. Default is './checkpoints'",
                        type=str,
                        default=DEFAULT_CHECKPOINTS_LOCATION)
    parser.add_argument("-d", "--dataset_location",
                        help="Location of the dataset. By default it will use the dataset included "
                             "in the challenge, copied at './data/",
                        type=str,
                        default=DEFAULT_DATASET_LOCATION)
    parser.add_argument("-w", "--workers",
                        help="The number of workers for the DataLoaders. Default is CPU_COUNT/2.",
                        type=int,
                        default=DEFAULT_NUM_WORKERS)
    return parser


if __name__ == "__main__":
    argument_parser = make_parser()
    args = argument_parser.parse_args()

    backbone_name = args.backbone_name
    max_epochs = args.max_epochs
    gpus = args.gpus
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoints_location = args.checkpoints_location
    dataset_location = args.dataset_location
    workers = args.workers

    run_train(
        backbone_name,
        checkpoints_location,
        learning_rate,
        batch_size,
        max_epochs,
        gpus,
        workers,
        dataset_location
    )



