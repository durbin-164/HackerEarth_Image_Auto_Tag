import os
import ast
import torch
from model_dispatcher import MODEL_DISPATCHER
from dataset import TestDataset
from torch import nn
import pandas as pd
from tqdm import tqdm

DEVICE = 'cuda'

IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH"))

TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))


def predict_to_numpy(predict):
    return torch.nn.functional.softmax(predict, dim=1).data.cpu().numpy().argmax(axis=1)


def get_prediction( data_loader, models):
    predictions = []
    total_model = float(len(models))

    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            image = d["image"]
            img_id = d["image_id"]
            image = image.to(DEVICE, dtype = torch.float)
            
            
            preds = torch.zeros(image.shape[0], 4, dtype = torch.float).to(DEVICE)
            
            for i in range(len(models)):
                p = models[i](image)
                preds += p/float(len(models))     
    
            preds = predict_to_numpy(preds)
            

            for ii, imid in enumerate(img_id):
                predictions.append((f"{imid}", preds[ii]))
        
    return predictions
                
def get_models():
    models =[]
    i =0
    for j in [0,1,2]:
        models.append(MODEL_DISPATCHER['resnet34'](pretrained=False))
        models[i].load_state_dict(torch.load(f'../save_model/resnet34_folds({j},).bin'))
        models[i].to(DEVICE)
        models[i].eval()
        i+=1
    return models

def save_submission(predictions):
    inverse_map={0:'Food',1:'Attire',2:'Decorationandsignage',3:'misc'}

    sub = pd.DataFrame(predictions, columns = ["Image", "label"])
    sub['Class']=sub['label'].map(inverse_map)
    sub = sub[['Image', 'Class']]
    sub.to_csv("../save_submission/submission.csv", index = False)
    print(sub.head(5))


def main():
    
    models = get_models()

    test_dataset  = TestDataset(
        img_height = IMAGE_HEIGHT,
        img_width = IMAGE_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False,
        num_workers = 4
    )

    predictions = get_prediction(test_loader, models)

    save_submission(predictions)
    
    

if __name__ == "__main__":
    main()


