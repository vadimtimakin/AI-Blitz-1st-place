import torch
import pandas as pd 
import os 
import random
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import models
from torch import nn
from albumentations.pytorch import ToTensor
from PIL import Image
from matplotlib import cm
import albumentations as A


def fullseed(seed=0xFACED):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class ImageDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""

    def __init__(self, images, transforms):
        self.images = images  # List with targets
        self.transforms = transforms  # Transforms

    def __getitem__(self, idx):
        inputs = list()
        src = cv2.imread(f"./data/train/Labels/{idx}.jpg")
        start = cv2.imread(os.path.join(os.path.join(PATH_TO_IMGS, str(idx)), "start_image.jpg"))
        final = cv2.imread(os.path.join(os.path.join(PATH_TO_IMGS, str(idx)), "last_image.jpg"))
        return transforms(image=src)["image"], start, final

    def __len__(self):
        return len(self.images)


fullseed()


NUMCLASSES = 256 * 256 * 3
PATH_TO_IMGS = "./data/test/Corrupted_Images"
DEVICE = "cuda"
MODELNAME = "mobilenet_v2"
PATH_TO_CHK = "./weigths/thebest.pt"

transforms = A.Compose([
    # A.RandomRotate90()

    # A.VerticalFlip(p=1.0),
    # A.HorizontalFlip(p=1.0)
])

imgs = [i for i in range(5000)]

dataset = ImageDataset(imgs, transforms)
dataloader = torch.utils.data.DataLoader(dataset,
                                        shuffle=False,
                                        batch_size=1,
                                        pin_memory=True,
                                        num_workers=16,
                                        persistent_workers=False)

model = getattr(models, MODELNAME)(pretrained=False)
lastlayer = list(model._modules)[-1]
try:
    setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer).in_features,
                                        out_features=NUMCLASSES, bias=True))
except torch.nn.modules.module.ModuleAttributeError:
    setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer)[1].in_features,
                                        out_features=NUMCLASSES, bias=True))
# model.load_state_dict(torch.load(PATH_TO_CHK)["model"])
# model.to(DEVICE)

preds = []
# model.eval()
with torch.no_grad():
    for i, img in enumerate(tqdm(dataloader)):
        src, start, final = img
        start = np.array(start.squeeze(0))
        final = np.array(final.squeeze(0))
        outputs = np.array(src.squeeze(0))
        color = outputs.mean(axis=0).mean(axis=0)
        # outputs[0:256, 0:512] = A.Resize(256, 512)(image=start)["image"]
        # outputs[256:512, 0:512] = A.Resize(256, 512)(image=final)["image"]

        # currupted_im = final
        # pp_width, pp_height, _ = currupted_im.shape
        # for _ in range(9):
        #     random_x = random.randint(0, 512 - pp_width)
        #     random_y = random.randint(0, 512 - pp_height)
        #     outputs[random_x:random_x+pp_width, random_y:random_y+pp_height] = currupted_im

        # print(outputs.shape, type(outputs))
        # outputs += 1
        # outputs[0:start.shape[0], 0:start.shape[1]] = start
        # outputs[512 - final.shape[0]:512, 512-final.shape[1]:512] = final
        # print(outputs[outputs==[0, 0, 0]])
        # outputs[outputs==[0, 0, 0]] = color
        # for k in range(outputs.shape[0]):
        #     for j in range(outputs.shape[1]):
        #         if (outputs[k][j] < [30, 30, 30]).all():
        #             outputs[k][j] = color
        # outputs = A.VerticalFlip(p=1.0)(image=outputs)["image"]
        # outputs = A.HorizontalFlip(p=1.0)(image=outputs)["image"]
        cv2.imwrite(f"./Labels/{i}.jpg", outputs)
