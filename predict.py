import glob
import os

import torch
import torchvision
from PIL import Image
import numpy as np

from wcid import WCID

overlay = True

save_dir = "results/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create the model
model = WCID(in_channels=3, n_classes=1)
# Load state_dict
model.load_state_dict(torch.load("runs/1/best_model.pth"))

val_img_path = "/home/s0559816/Desktop/railway-segmentation/data/img/val/"
val_img = glob.glob(f"{val_img_path}*.png")

for image_path in val_img:
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    inferred_name = f"{image_name}_inferred.png"
    inferred_save_path = os.path.join(save_dir, inferred_name)

    image = Image.open(image_path)

    # Transform
    transforms = torchvision.transforms.ToTensor()
    input = transforms(image)
    input = torch.unsqueeze(input, dim=0)

    # Set model to eval
    model.eval()
    with torch.no_grad():
        output = model(input)

    output = output.cpu().numpy()
    output_array = np.squeeze(output, axis=(0, 1))
    output_array = output_array > 0.5

    inferred_image = Image.fromarray(output_array).convert("L")

    if inferred_image.width != image.width or inferred_image.height != image.height:
        inferred_image.resize((image.width, image.height), resample=Image.BICUBIC)

    inferred_image.save(inferred_save_path, quality=100, subsampling=0)

    if overlay:
        overlayed_name = f"{image_name}_overlayed.png"
        overlayed_save_path = os.path.join(save_dir, inferred_name)

        overlayed = Image.blend(image, inferred_image.convert("RGB"), alpha=0.5)
        overlayed.save(overlayed_save_path, quality=100, subsampling=0)
