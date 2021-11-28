import glob
import os

import torch
import torchvision
from PIL import Image
import numpy as np
from matplotlib.pyplot import get_cmap

from wcid import WCID

overlay = True

save_dir = "results/"
inferred_dir = os.path.join(save_dir, "inferred")
overlay_dir = os.path.join(save_dir, "overlayed")

for dir in [save_dir, inferred_dir, overlay_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create the model
model = WCID(in_channels=3, n_classes=1)
# Load state_dict
model.load_state_dict(torch.load("runs/10/best_model.pth"))

val_img_path = "/home/s0559816/Desktop/railway-segmentation/data/img/val/"
val_img = glob.glob(f"{val_img_path}*.png")

for image_path in val_img:
    image = Image.open(image_path)

    # Transform
    # transforms = torchvision.transforms.ToTensor()
    input = torchvision.transforms.ToTensor()(image)
    input = torch.unsqueeze(input, dim=0)

    # Set model to eval
    model.eval()
    with torch.no_grad():
        output = model(input)

    output = output.squeeze(0)

    colormap = get_cmap("jet")
    out_image_arr = colormap(output)[:, :, :3] * 255
    out_image_arr = out_image_arr.astype(np.uint8)

    # inferred_image = torchvision.transforms.ToPILImage()(output.squeeze(0))
    inferred_image = Image.fromarray(out_image_arr)

    if inferred_image.width != image.width or inferred_image.height != image.height:
        inferred_image.resize((image.width, image.height), resample=Image.BICUBIC)

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    inferred_name = f"{image_name}_inferred.png"
    inferred_save_path = os.path.join(inferred_dir, inferred_name)

    inferred_image.save(inferred_save_path, quality=100, subsampling=0)

    if overlay:
        overlayed_name = f"{image_name}_overlayed.png"
        overlayed_save_path = os.path.join(overlay_dir, overlayed_name)

        overlayed = Image.blend(image, inferred_image.convert("RGB"), alpha=0.5)
        overlayed.save(overlayed_save_path, quality=100, subsampling=0)
