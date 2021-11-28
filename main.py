import os
import datetime
from multiprocessing import cpu_count
import logging

from cv2 import transform
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

from dataset import RailData
from wcid import NetSeq
from wcid import WCID
from p_logging import val_logging
from validation.metrics import calculate_metrics


def run(
    train_img,
    train_msk,
    val_img,
    val_msk,
    resolution=(320, 160),
    epochs=5,
    bs=1,
    lr=1e-3,
    weights_pth=None,
    save_path=None,
):
    """

    :param train_img: Path to training images.
    :param train_msk: Path to training masks.
    :param val_img: Path to validation images.
    :param val_msk: Path to validation masks.
    :param res_scale: Scale height and width of image.
    :param epochs: Training epochs.
    :param bs: Batch size.
    :param lr: Learning rate
    :param weights_pth: Path to weights from previous training.
    :return: None.
    """
    # Training start time
    start_datetime = datetime.datetime.now()
    writer = SummaryWriter(save_path)

    log_path = os.path.join(save_path, "loggs/")
    weighs_path = os.path.join(save_path, "weights/")

    for dir in [log_path, weighs_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    logging.basicConfig(
        level=logging.INFO, filename=os.path.join(save_path, "loggs/log.txt")
    )

    # Computing device
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Instance of neural network
    # net = NetSeq()
    net = WCID(in_channels=3, n_classes=1)
    net = net.to(dev)
    # Prepare data parallel
    # net = nn.DataParallel(net)

    # Load weights
    if weights_pth is not None:
        net.load_state_dict(torch.load(weights_pth, map_location=dev))
        weight_file_name = os.path.basename(weights_pth)
        weight_file_name = os.path.splitext(weight_file_name)[-2]
        start_epoch = int(weight_file_name.replace("CP_epoch", ""))
        print(f"Continue training in epoch {start_epoch + 1}")
    else:
        start_epoch = 0

    # Training and validation Dataset
    train_dataset = RailData(train_img, train_msk, resolution, transform=True)
    val_dataset = RailData(val_img, val_msk, resolution)

    # Length of training and validation Dataset
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # Create data loader
    cpus = cpu_count()
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=cpus,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=cpus,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer and learning rate scheduler
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9)  # weight_decay=1e-8
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, verbose=True
    )

    # Loss function (binary cross entropy)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()

    overall_batches = 0
    last_val_loss = float("inf")
    best_IoU = 0
    # Training loop
    for epoch in range(start_epoch, epochs + start_epoch):
        net.to(dev)
        net.train()
        train_loss = 0.0
        desc = f"Epoch {epoch + 1}/{epochs}"
        # Epoch progress bar
        with tqdm(total=n_train, desc=desc, leave=False, position=0) as bar:
            # Training batches
            for batch in train_loader:
                # Increment bar by batch size
                bar.update(bs)

                # Get images from batch
                images = batch["image"]
                masks = batch["mask"]

                # Load images and masks to computing device
                images = images.to(device=dev, dtype=torch.float32)
                masks = masks.to(device=dev, dtype=torch.float32)

                # Clear old gradients and loss backpropagation
                optimizer.zero_grad()

                # Predict masks from images
                prediction = net(images)

                # Calculate loss
                loss = criterion(prediction, masks)
                writer.add_scalar("Loss/train", loss, epoch)

                loss.backward()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)  # Why???
                optimizer.step()

                # Accumulate batch loss to epoch loss
                train_loss += loss.item()

                # Increase batches counter
                overall_batches += 1

        valid_loss = 0.0
        iou, f1, acc, pre, rec = 0, 0, 0, 0, 0

        net.eval()
        # torch.cuda.empty_cache()
        for batch in val_loader:
            images = batch["image"]
            masks = batch["mask"]

            # Load images and masks to computing device
            images = images.to(device=dev, dtype=torch.float32)
            masks = masks.to(device=dev, dtype=torch.float32)

            with torch.no_grad():
                prediction = net(images)

            loss = criterion(prediction, masks)
            writer.add_scalar("Loss/val", loss, epoch)
            valid_loss += loss.item()

            # Threshold at 0.5 between 0 and 1
            prediction = prediction > 0.5

            metrics = calculate_metrics(prediction, masks)
            iou += metrics["iou"]
            f1 += metrics["f1"]
            acc += metrics["accuracy"]
            pre += metrics["precision"]
            rec += metrics["recall"]

        # Normalize Validation metrics
        valid_loss /= n_val
        iou /= n_val
        f1 /= n_val
        acc /= n_val
        pre /= n_val
        rec /= n_val

        writer.add_scalar("metrics/IoU", iou, epoch)
        writer.add_scalar("metrics/Accuracy", acc, epoch)

        # Validation message
        val_msg = f"Epoch {epoch+1}\n"
        val_msg += f"Training Loss: {train_loss / len(train_loader)}\tValidation Loss: {valid_loss} {(Fore.RED + '↑') if valid_loss > last_val_loss else (Fore.GREEN +'↓')+Fore.RESET}\n"
        val_msg += f"Validation Scores:\n"
        val_msg += f"   IoU:       {iou:.3f}\tF1:     {f1:.3f}\taccuracy: {acc:.3f}\n"
        val_msg += f"   precision: {pre:.3f}\trecall: {rec:.3f}\n"

        print(val_msg)
        logging.info(val_msg)

        # Validation logg
        logg_file_pth = os.path.join(log_path, f"{start_datetime.isoformat()}.csv")
        val_logging.val_metrics_logger(metrics, logg_file_pth)

        last_val_loss = valid_loss

        if best_IoU < iou:
            torch.save(net.state_dict(), os.path.join(save_path, f"best_model.pth"))
            best_IoU = iou

        # Saving State Dict
        torch.save(
            net.state_dict(), os.path.join(weighs_path, f"CP_epoch{epoch + 1}.pth")
        )

        scheduler.step(train_loss / n_train)

    hparams_dict = {
        "lr": lr,
        "batch_size": bs,
        "epochs": epochs,
        "optimizer": optimizer.__class__.__name__,
        "loss": criterion.__class__.__name__,
    }

    writer.flush()
    writer.close()


def main():
    colorama.init(autoreset=True)

    # train_img = "/home/s0559816/Desktop/mono/images/trn/"
    # train_msk = "/home/s0559816/Desktop/mono/masks/trn/"

    # val_img = "/home/s0559816/Desktop/mono/images/val/"
    # val_msk = "/home/s0559816/Desktop/mono/masks/val/"

    train_img = "/home/s0559816/Desktop/railway-segmentation/data/img/trn"
    train_msk = "/home/s0559816/Desktop/railway-segmentation/data/msk/trn"

    val_img = "/home/s0559816/Desktop/railway-segmentation/data/img/val"
    val_msk = "/home/s0559816/Desktop/railway-segmentation/data/msk/val"
    weights_pth = "runs/3/weights/CP_epoch50.pth"

    save_path = "runs"

    list_subfolders = [f.path for f in os.scandir(save_path) if f.is_dir()]
    list_subfolders.sort()

    head, tail = os.path.split(list_subfolders[-1])
    new_tail = int(tail) + 1
    save_path = os.path.join(head, str(new_tail))

    os.makedirs(save_path, exist_ok=True)

    run(
        train_img,
        train_msk,
        val_img,
        val_msk,
        resolution=(320, 160),
        epochs=100,
        bs=1,
        lr=1e-0,
        weights_pth=weights_pth,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
