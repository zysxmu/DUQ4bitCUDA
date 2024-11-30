import os
import torch
import math
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class RandomNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=(3,224,224), num_classes=1000, length=1000):
        self.size = img_size
        self.num_classes = num_classes
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        fake_img = torch.randn(self.size)
        fake_label = torch.randint(0, self.num_classes, (1,)).item()
        return fake_img, fake_label

def build_randval_dataset(args):  
    val_dataset = RandomNoiseDataset(args.fake_image_size, args.fake_num_classes, args.fake_data_len)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return val_dataloader
    

def build_dataset(args):
    model_type = args.model.split("_")[0]
    if model_type == "deit":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif model_type == 'vit':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif model_type == 'swin':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    # Data
    traindir = os.path.join(args.dataset, 'train')
    valdir = os.path.join(args.dataset, 'val')

    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.calib_batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return train_loader, val_loader


def build_transform(input_size=224, interpolation="bicubic",
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):
    def _pil_interp(method):
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            return Image.BILINEAR
    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size, interpolation=ip
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
