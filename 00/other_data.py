import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
class mydataset(Dataset):
    def __init__(self, csv_path, image_dir,has_label=True):
        self.classes=["invoice","receipt","schedule","poster","lab_note","notice","handwritten_note","form",
         "meeting_minutes","grade_report"]
        self.has_label=has_label
        self.df=pd.read_csv(csv_path)
        self.image_dir=image_dir
        self.label_map={name:idx for idx,name in enumerate(self.classes)}
        self.idx_to_label={idx: name for name,idx in self.label_map.items()}
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        filename=self.df.iloc[idx]["image"]
        image_path=filename
        image=Image.open(image_path).convert("RGB")
        image=self.transform(image)
        if self.has_label:
            label_name = self.df.iloc[idx]["label"]
            label=self.label_map[label_name]
            return image, label
        else:
            sample_id = self.df.iloc[idx]["id"]
            return image, sample_id
#dataset=mydataset("train.csv","train")
#image,label=dataset[0]
#print(image.shape)
#print(label)

