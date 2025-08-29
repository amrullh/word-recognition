import pandas as pd
import torch
import albumentations
import numpy as np
from PIL import Image, ImageFile
import string
import os   
from config import  DATA_DIR, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, EPOCHS, LEARNING_RATE, NUM_WORKERS, DEVICE

ImageFile.LOAD_TRUNCATED_IMAGES = True

class dataset:
    def __init__(self, image_path, targets, resize=None):
        self.image_path = image_path
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])
        
        # mapping huruf ke angka
        alphabet = string.ascii_lowercase   # 'abcdefghijklmnopqrstuvwxyz'
        self.char2idx = {ch: idx+1 for idx, ch in enumerate(alphabet)}  # mulai dari 1
        self.char2idx["<pad>"] = 0  # buat padding jika perlu

    def __len__(self):
        return len(self.image_path)
    
    def text_to_seq(self, text):
        """Ubah string label jadi list angka huruf per huruf"""
        return [self.char2idx[c] for c in text.lower() if c in self.char2idx]

    def __getitem__(self, item):
        # ambil gambar
        image = Image.open(self.image_path[item]).convert("RGB")
        targets = self.targets[item]   # string kata, misal "kucing"

        if self.resize is not None:
            image = image.resize((self.resize[0], self.resize[1]), resample=Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # ubah kata jadi list angka
        target_seq = self.text_to_seq(targets)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target_seq, dtype=torch.long),   # urutan huruf dalam bentuk tensor
            "target_length": torch.tensor(len(target_seq), dtype=torch.long)  # panjang kata
        }


