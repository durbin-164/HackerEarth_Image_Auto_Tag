import pandas as pd 
import joblib
from PIL import Image
import albumentations
import numpy as np
import torch

class TrainDataset:
    def __init__(self,folds, img_height, img_width, mean, std):
        df = pd.read_csv('../input/dataset/train_folds.csv')
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
                
            
                albumentations.OneOf([
                    albumentations.ShiftScaleRotate(shift_limit=0.0625, 
                                                scale_limit=0.1,
                                                rotate_limit=45),
                    albumentations.Rotate(limit = 5),
                    albumentations.RandomGamma(),
                    albumentations.RandomShadow(),
                    albumentations.RandomGridShuffle(),
                    albumentations.ElasticTransform(),
                    albumentations.RGBShift(),
                ])  ,
                
                albumentations.OneOf([

                    albumentations.OneOf([
                        albumentations.Blur(),
                        albumentations.MedianBlur(),
                        albumentations.MotionBlur(),
                        albumentations.GaussianBlur(),

                    ]),

                    albumentations.OneOf([
                        albumentations.GaussNoise(),
                        albumentations.IAAAdditiveGaussianNoise(),
                        albumentations.ISONoise()
                    ]),

                ]),

                albumentations.OneOf([
                    albumentations.RandomBrightness(),
                    albumentations.RandomContrast(),
                    albumentations.RandomBrightnessContrast(),
                ]),

               

                albumentations.OneOf([ 
                    albumentations.OneOf([
                        albumentations.Cutout(),
                        albumentations.CoarseDropout(),
                        albumentations.GridDistortion(),
                        albumentations.GridDropout(),
                        albumentations.OpticalDistortion()

                    ]),
                    

                    albumentations.OneOf([
                        albumentations.HorizontalFlip(),
                        albumentations.VerticalFlip(),
                        albumentations.RandomRotate90(),
                        albumentations.Transpose()
                    ]),

                ]),

                

        
                # albumentations.OneOf([
                #         albumentations.RandomSnow(),
                #         albumentations.RandomRain(),
                #         albumentations.RandomFog(),
                #     ]),

  
                albumentations.Normalize(mean, std, always_apply = True)
            ])


    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, item):
        
        image = Image.open(f'../input/dataset/Train Images/{self.image_ids[item]}')

        

        #image = image.reshape(137,236).astype(float)
        image = image.convert("RGB")
        image = self.aug(image = np.array(image))["image"]

        image = np.transpose(image, (2,0,1)).astype(np.float32)

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

        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return {
            'image': torch.tensor(image, dtype = torch.float),
            'image_id':image_id
        }

