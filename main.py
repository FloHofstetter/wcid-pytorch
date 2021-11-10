from dataset import RailData
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from multiprocessing import cpu_count
import pathlib
from tqdm import tqdm
from wcid import NetSeq
import sys
from validation.metrics import calculate_metrics
import os
import colorama
from colorama import Fore, Back, Style
from p_logging import val_logging
from torchsummary import summary
from torchvision import datasets
import datetime


def train(
    train_img,
    train_msk,
    val_img,
    val_msk,
    res_scale=0.1,
    epochs=5,
    bs=1,
    lr=1e-3,
    weights_pth=None,
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

    # Computing device
    # dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev = "cpu"

    # Instance of neural network
    net = NetSeq()
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
    train_dataset = RailData(train_img, train_msk, res_scale, transform=True)
    val_dataset = RailData(val_img, val_msk, res_scale)

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
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
         optimizer, "max", patience=100, verbose=True
    )

    # Loss function (binary cross entropy)
    criterion = nn.BCEWithLogitsLoss()

    overall_batches = 0
    last_val_loss = float("inf")
    # Training loop
    for epoch in range(start_epoch, epochs + start_epoch):
        net.to(dev)
        net.train()
        epoch_loss = 0
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
                # print(f"{images.device=}")
                # print(f"{masks.device=}")
                # print(f"{next(net.parameters()).device=}")

                # Predict masks from images
                prediction = net(images)

                # Calculate loss
                loss = criterion(prediction, masks)

                # Accumulate batch loss to epoch loss
                epoch_loss += loss.item()

                # Clear old gradients and loss backpropagation
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)  # Why???
                optimizer.step()

                # Increase batches counter
                overall_batches += 1

                # Validate 10 times per epoch with validation set
                if False:   # overall_batches % (n_train // (10 * bs)) == 0:
                    val_loss = 0
                    iou, f1, acc, pre, rec = 0, 0, 0, 0, 0
                    # Set neural net to evaluation state
                    net.eval()
                    for val_batch in val_loader:

                        # Get images from batch
                        images = val_batch["image"]
                        masks = val_batch["mask"]

                        # Load images and masks to computing device
                        images = images.to(device=dev, dtype=torch.float32)
                        masks = masks.to(device=dev, dtype=torch.float32)

                        # Predict validation batch (no gradients needed)
                        with torch.no_grad():
                            prediction = net(images)

                        # Calculate validation loss
                        criterion = nn.BCEWithLogitsLoss()

                        # Validation loss
                        loss = criterion(prediction, masks)
                        val_loss += loss

                        # Force prediction between 0 and 1
                        # prediction = torch.sigmoid(prediction)

                        # Threshold at 0.5 between 0 and 1
                        prediction = prediction > 0.5

                        # TODO: Validation metrics
                        metrics = calculate_metrics(prediction, masks)
                        iou += metrics["iou"]
                        f1 += metrics["f1"]
                        acc += metrics["acc"]
                        pre += metrics["pre"]
                        rec += metrics["rec"]

                    # Normalize Validation metrics
                    val_loss /= n_val
                    iou /= n_val
                    f1 /= n_val
                    acc /= n_val
                    pre /= n_val
                    rec /= n_val

                    # Validation message
                    sys.stdout.write("\r\033[K")
                    val_msg = f"    Validated with "
                    val_msg += f"IoU: {iou:.1f} F1: {f1:.2f} ACC: {acc:.2f}"
                    val_msg += f" Pre: {pre:.2f} Rec: {rec:.2f}"
                    val_msg += f" Lss: {val_loss:.3e} ✓"
                    val_msg += f" {(Fore.RED + '↑') if val_loss > last_val_loss else (Fore.GREEN +'↓')}"
                    last_val_loss = val_loss
                    print(val_msg)

                    # Validation logg
                    logg_file_pth = os.path.join(
                        "loggs/", f"{start_datetime.isoformat()}.csv"
                    )
                    val_logging.val_metrics_logger(metrics, logg_file_pth)

        scheduler.step(epoch_loss / n_train)
        epoch_msg = (
            f"Trained epoch {epoch + 1:02d} with loss {epoch_loss / n_train:.3e} "
        )
        epoch_msg += f"at learning rate {optimizer.param_groups[0]['lr']:.3e} ✓"
        print(epoch_msg)

        # Save weights every epoch
        weight_pth = "weight/"
        pathlib.Path(weight_pth).mkdir(parents=True, exist_ok=True)
        net.to("cpu")
        torch.save(net.state_dict(), weight_pth + f"CP_epoch{epoch + 1}.pth")
        net.to(dev)


def main():
    colorama.init(autoreset=True)

    # """
    train_img = "/media/flo/External/files_not_unter_backup/nlb/smr/nlb_summer/img_h/trn_0/"
    train_msk = "/media/flo/External/files_not_unter_backup/nlb/smr/nlb_summer/msk_track_bin/png_uint8_h/trn_0/"
    val_img = "/media/flo/External/files_not_unter_backup/nlb/smr/nlb_summer/img_h/val_0/"
    val_msk = "/media/flo/External/files_not_unter_backup/nlb/smr/nlb_summer/msk_track_bin/png_uint8_h/val_0/"
    weights_pth = None  # "weight/CP_epoch26.pth"
    train(
        train_img,
        train_msk,
        val_img,
        val_msk,
        res_scale=0.2,
        epochs=80000,
        bs=1,
        lr=1e-0,
        weights_pth=weights_pth,
    )
    """

    model = NetSeq()
    summary(model, (3, 160, 320), device="cpu", col_names=["input_size", "output_size", "num_params"])
    """


if __name__ == "__main__":
    main()
