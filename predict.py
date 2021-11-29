import glob
import os
import PIL

import torch
import torchvision
import torchsummary 
from PIL import Image
import numpy as np
from matplotlib.pyplot import get_cmap

from wcid import WCID

overlay = True
n_classes = 1

save_dir = "results/"
inferred_dir = os.path.join(save_dir, "inferred")
overlay_dir = os.path.join(save_dir, "overlayed")

for dir in [save_dir, inferred_dir, overlay_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create the model
model = WCID(in_channels=3, n_classes=n_classes)
# Load state_dict
model.load_state_dict(torch.load("runs/1/best_model.pth"))
# torchsummary.summary(model=model, input_size=(3, 1000, 2000))

model.eval()

val_img_path = "/home/s0559816/Desktop/railway-segmentation/data/img/val/"
# val_img_path = "/home/s0559816/Desktop/mono/images/val/"
val_img = glob.glob(f"{val_img_path}*.png")

if n_classes > 2:
    outputs = {}

for image_path in val_img:
    image = Image.open(image_path)

    # Transform
    # transforms = torchvision.transforms.ToTensor()
    input = torchvision.transforms.ToTensor()(image)
    input = torch.unsqueeze(input, dim=0)

    # Set model to eval
    with torch.no_grad():
        output = model(input)
        output = torch.sigmoid(output)

    # post process output
    # output = torch.nn.functional.sigmoid(output)
    if n_classes <= 1:
        output = output.cpu().numpy() 
        output = np.squeeze(output) 
        output = output > 0.5
        output = (output * 255).astype(np.uint8)

        inferred_image = Image.fromarray(output)
    else:
        output_data = torch.nn.functional.softmax(output, dim=1).cpu().data
        max_probs, predictions = output_data.max(1)
        outputs['predictions'] = predictions
        outputs['prob_mask'] = max_probs

        inferred_image = Image.fromarray(predictions)

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