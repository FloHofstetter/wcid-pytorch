import torch
from torchmetrics.functional import iou
import torchvision
from PIL import Image
from wcid import WCID

image = "/home/s0559816/Desktop/mono/images/val/lrc_data_20200520_083044_all.mp4_00014000.png"
label = "/home/s0559816/Desktop/mono/masks/val/lrc_data_20200520_083044_all.mp4_00014000.png"
label = Image.open(label)
image = Image.open(image)

transforms = torchvision.transforms.ToTensor()
label = transforms(label).type(torch.IntTensor)
input = transforms(image)

# network expects a batch, this adds the batch dimension
input = torch.unsqueeze(input, dim=0)

model = WCID(in_channels=3, n_classes=1)
model.load_state_dict(torch.load("runs/1/best_model.pth"))
model.eval()
with torch.no_grad():
    prediction = model(input)
prediction = prediction > 0.5
# remove batch dimension
prediction = prediction.squeeze(0)

print(iou(prediction, label).item())
