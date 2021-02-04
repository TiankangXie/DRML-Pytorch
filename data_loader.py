import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
# https://github.com/omarsayed7/Deep-Emotion/blob/master/data_loaders.py
GEN_PATH = "F:\\FaceExprDecode\\aligned\\"
class image_Loader(Dataset):
    def __init__(self, csv_dir, img_dir, transform=None):
        self.main_csv = pd.read_csv(csv_dir)
        self.img_dir = img_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(170),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform


    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir,str(self.main_csv["Subject"][index]),str(self.main_csv["Task"][index]),str(self.main_csv["Number"][index])+".jpg")
        img = Image.open(img_name)
        img = self.transform(img)
        aus = self.main_csv[["1","2","4","6","7","10","12","14","15","17","23","24"]]
        aus = aus.to_numpy(dtype='int')
        return(img, aus[index,:])

    def __len__(self) -> int:
        return self.main_csv.shape[0]
    