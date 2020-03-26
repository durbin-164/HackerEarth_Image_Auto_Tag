import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
import albumentations
import joblib
import glob
import os
from tqdm import tqdm


def augment(aug, image):
    return aug(image=image)['image']

def h_flip_blur(p=1.0):
    return albumentations.Compose([
        albumentations.HorizontalFlip(p=p),
        albumentations.Blur(p=p),
        albumentations.GaussNoise(p=p)
    ], p=p)

def v_flib_g_blur(p=1.0):
    return albumentations.Compose([
        albumentations.VerticalFlip(p=p),
        albumentations.GaussianBlur(p=p),
        albumentations.IAAAdditiveGaussianNoise(p=p)
    ], p=p)

def r_ss_roate_m_blur(p=1.0):
    return albumentations.Compose([
        albumentations.ShiftScaleRotate(p=p),
        albumentations.MedianBlur(p=p),
        albumentations.MultiplicativeNoise(p=p)
    ], p=p)

def rgd_shuffle_mo_blur(p=1.0):
    return albumentations.Compose([
        albumentations.ChannelShuffle(p=p),
        albumentations.MotionBlur(p=p),
        albumentations.RandomBrightness(p=1)
    ], p=p)


def c_crop_g_blur(i_height=256, i_width=256,p=1.0):
    return albumentations.Compose([
        albumentations.CenterCrop(height=150, width=150,p=p),
        albumentations.Resize(i_height, i_width),
        albumentations.GlassBlur(p=p),
        albumentations.RandomContrast(p=p)
    ], p=p)

def r_crop_clahe(i_height=256, i_width=256,p=1.0):
    return albumentations.Compose([
        albumentations.RandomCrop(height=150, width=150,p=p),
        albumentations.Resize(i_height, i_width),
        albumentations.CLAHE(p=p),
        albumentations.RandomBrightnessContrast(p=p)
    ], p=p)

def elastic_tranform_r_brightness(p=1.0):
    return albumentations.Compose([
        albumentations.ElasticTransform(p=p),
        albumentations.RandomBrightnessContrast(p=p),
        albumentations.RandomRain(p=p)
    ], p=p)

def pre_porcess(height=256, width=256, p=1.0):
    return albumentations.Compose([
        albumentations.Resize(height, width)
    ], p=p)


if __name__ == "__main__":

    train = pd.read_csv('../input/dataset/train_folds.csv')

    files = glob.glob('../input/dataset/Train Images/*')
    
    dataset =[]

    counter=0

    for j, file in tqdm(enumerate(files), total= len(files)):
        file_name = os.path.basename(file)
        data_row = train[train.Image==file_name]
        label = data_row.label.iloc[0]
        kfold = data_row.kfold.iloc[0]

        image = Image.open(file)
        image = image.convert("RGB")

        #original image
        aug = pre_porcess(256,256)
        image = augment(aug, np.array(image))
        new_file_name = f"image{counter}"
        counter+=1
        joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
        dataset.append((new_file_name, label, kfold))

        #augment -1
        aug = h_flip_blur(p=1.0)
        image = augment(aug, np.array(image))
        new_file_name = f"image{counter}"
        counter+=1
        joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
        dataset.append((new_file_name, label, kfold))

        #augment -2
        aug = v_flib_g_blur(p=1.0)
        image = augment(aug, np.array(image))
        new_file_name = f"image{counter}"
        counter+=1
        joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
        dataset.append((new_file_name, label, kfold))

        #augment -3
        aug = r_ss_roate_m_blur(p=1.0)
        image = augment(aug, np.array(image))
        new_file_name = f"image{counter}"
        counter+=1
        joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
        dataset.append((new_file_name, label, kfold))

        
        if label in ['misc', 'Attire', 'Decorationandsignage']:
            #augment -4
            aug = rgd_shuffle_mo_blur(p=1.0)
            image = augment(aug, np.array(image))
            new_file_name = f"image{counter}"
            counter+=1
            joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
            dataset.append((new_file_name, label, kfold))

        if label in [ 'misc', 'Decorationandsignage']:
            #augment -5
            aug = c_crop_g_blur(i_height=256, i_width=256,p=1.0)
            image = augment(aug, np.array(image))
            new_file_name = f"image{counter}"
            counter+=1
            joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
            dataset.append((new_file_name, label, kfold))

        if label in ['Decorationandsignage']:
            #augment -6
            aug = r_crop_clahe(i_height=256, i_width=256,p=1.0)
            image = augment(aug, np.array(image))
            new_file_name = f"image{counter}"
            counter+=1
            joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
            dataset.append((new_file_name, label, kfold))

            #augment -7
            aug = elastic_tranform_r_brightness(p=1.0)
            image = augment(aug, np.array(image))
            new_file_name = f"image{counter}"
            counter+=1
            joblib.dump(image, f"../input/image_pickles/{new_file_name}.pkl")
            dataset.append((new_file_name, label, kfold))


    print(counter)

    train_new = pd.DataFrame(dataset, columns = ["Image", "label", "kfold"])
    train_new.to_csv("../input/train_folds_new.csv", index = False)