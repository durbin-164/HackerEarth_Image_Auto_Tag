import pandas as pd
from sklearn.model_selection import KFold

if __name__ == "__main__":
    df = pd.read_csv('../input/dataset/train_mod.csv')
    #print(df.head())

    df.loc[:, 'kfold'] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    X = df.Image.values
    y = df.label.values

    kfold = KFold(3, True, 1)


    for fold, (trn_, val_) in enumerate(kfold.split(X,y)):
        print("Train: ", trn_, "Val: ", val_)
        
        df.loc[val_, 'kfold'] = fold

    #print(df.kfold.value_counts)
    df.to_csv("../input/dataset/train_folds.csv", index = False)


