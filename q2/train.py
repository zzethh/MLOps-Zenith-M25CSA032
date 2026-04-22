import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import CityScapesDataset
from model import UNet

IMG_DIR = "data/CameraRGB"
MASK_DIR = "data/CameraMask"
NUM_CLASSES = 23
EPOCHS = 15
BATCH_SIZE = 4
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

img_names = sorted(os.listdir(IMG_DIR))
mask_names = sorted(os.listdir(MASK_DIR))
img_paths = [os.path.join(IMG_DIR, n) for n in img_names]
mask_paths = [os.path.join(MASK_DIR, n) for n in mask_names]

dataset = CityScapesDataset(img_paths, mask_paths)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = UNet(NUM_CLASSES).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def compute_metrics(pred, mask, n_classes=NUM_CLASSES):
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    dice_list = []
    for c in range(n_classes):
        pred_c = (pred == c)
        mask_c = (mask == c)
        intersection = (pred_c & mask_c).float().sum().item()
        union = (pred_c | mask_c).float().sum().item()
        pred_area = pred_c.float().sum().item()
        mask_area = mask_c.float().sum().item()
        if union > 0:
            iou_list.append(intersection / union)
        if (pred_area + mask_area) > 0:
            dice_list.append(2.0 * intersection / (pred_area + mask_area))
    miou = np.mean(iou_list) if iou_list else 0.0
    mdice = np.mean(dice_list) if dice_list else 0.0
    return miou, mdice

log_file = open("training_log.txt", "w")

losses_list = []
miou_list = []
dice_list = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    count = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        loss = loss_fn(out, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        miou, mdice = compute_metrics(out.detach(), mask)
        total_loss += loss.item()
        total_iou += miou
        total_dice += mdice
        count += 1

    avg_loss = total_loss / count
    avg_iou = total_iou / count
    avg_dice = total_dice / count

    losses_list.append(avg_loss)
    miou_list.append(avg_iou)
    dice_list.append(avg_dice)

    msg = f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | mIOU: {avg_iou:.4f} | mDice: {avg_dice:.4f}"
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

torch.save(model.state_dict(), "saved_model.pth")
print("Model saved")
log_file.write("Model saved\n")

model.eval()
test_iou = 0
test_dice = 0
test_count = 0
with torch.no_grad():
    for img, mask in test_loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        miou, mdice = compute_metrics(out, mask)
        test_iou += miou
        test_dice += mdice
        test_count += 1

test_miou = test_iou / test_count
test_mdice = test_dice / test_count

msg = f"\nTest mIOU: {test_miou:.4f}\nTest mDice: {test_mdice:.4f}"
print(msg)
log_file.write(msg + "\n")

with open("test_scores.txt", "w") as f:
    f.write(f"{test_miou:.4f}\n{test_mdice:.4f}\n")

os.makedirs("plots", exist_ok=True)

plt.figure()
plt.plot(range(1, EPOCHS+1), losses_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.savefig("plots/loss.png")
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS+1), miou_list, marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("mIOU")
plt.title("Training mIOU")
plt.grid(True)
plt.savefig("plots/miou.png")
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS+1), dice_list, marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("mDice")
plt.title("Training mDice")
plt.grid(True)
plt.savefig("plots/dice.png")
plt.close()

log_file.write("Plots saved to plots/\n")
log_file.close()
print("Plots saved to plots/")
print("Done")