import pandas as pd 
import joblib
from PIL import Image
import albumentations
import numpy as np
import torch

class TrainDataset:
    def __init__(self,folds, img_height, img_width, mean, std):
        df = pd.read_csv('../input/train_folds.csv')
        df = df[['Image','label', 'kfold']]

        df = df[df.kfold.isin(folds)].reset_index(drop = True)

        self.image_ids = df.Image.values

        self.labels = df.label.values

        if len(folds)==1:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width),
                albumentations.Normalize(mean, std, always_apply = True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, 
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p = 0.9),
                
                albumentations.Rotate(limit = 5),
                albumentations.RandomGamma(),
                albumentations.RandomShadow(),
                albumentations.RandomGridShuffle(),
                albumentations.Cutout(),
                albumentations.CoarseDropout(),
                albumentations.GridDistortion(),

                albumentations.Normalize(mean, std, always_apply = True)
            ])


    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, item):
        
        image = joblib.load(f'../input/image_pickles/{self.image_ids[item]}.pkl')

        

        #image = image.reshape(137,236).astype(float)
        #image = image.convert("RGB")
        image = self.aug(image = np.array(image))["image"]

        image = np.transpose(image, (2,0,1)).astype(np.float32)/255.0

        return {
            'image': torch.tensor(image, dtype = torch.float),
            'label': torch.tensor(self.labels[item], dtype = torch.long),
        }



class TestDataset:
    def __init__(self,img_height, img_width, mean, std):
        df = pd.read_csv('../input/dataset/test.csv')
        #df = df[['Image','label', 'kfold']]

        #df = df[df.kfold.isin(folds)].reset_index(drop = True)

        self.image_ids = df.Image.values

        self.aug = albumentations.Compose([
            albumentations.Resize(img_height, img_width),
            albumentations.Normalize(mean, std, always_apply = True)
        ])
    


    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, item):
        
        image = Image.open(f'../input/dataset/Test Images/{self.image_ids[item]}')
        image_id = self.image_ids[item]
        #image = image.reshape(137,236).astype(float)
        image = image.convert("RGB")
        image = self.aug(image = np.array(image))["image"]

        image = np.transpose(image, (2,0,1)).astype(np.float32)/255.0

        return {
            'image': torch.tensor(image, dtype = torch.float),
            'image_id':image_id
        }

