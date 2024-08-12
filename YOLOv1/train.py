import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import VOCDataset
from model import YOLOv1
from utils import (
    intersection_over_union, 
    non_max_suppression,
    mean_average_precision,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
IMG_SIZE = 448
NUM_CLASSES = 20
CONF_THRESHOLD = 0.5
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = 7


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
    

transforms = Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

    return sum(mean_loss)/len(mean_loss)


def main():
    model = YOLOv1(split_size=S, num_boxes=5, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "E:/Documents/Python Scripts/Computer Vision/YOLOv1/data/8examples.csv", transform=transforms,
        img_dir=IMG_DIR, label_dir=LABEL_DIR
    )
    test_dataset = VOCDataset(
        "E:/Documents/Python Scripts/Computer Vision/YOLOv1/data/100examples.csv", transform=transforms,
        img_dir=IMG_DIR, label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()