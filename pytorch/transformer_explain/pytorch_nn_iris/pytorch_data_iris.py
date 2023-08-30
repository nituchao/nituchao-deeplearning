import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing

##############################################################################
# 数据加载
##############################################################################
dataset_iris_train = None
dataset_iris_test = None
class IrisDataSet(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def dataset_parser(filepath='', train_split=0.75):
    dataset = pd.read_csv(filepath, header=0, usecols=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
    dataset_features = dataset[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
    
    label_encoder=preprocessing.LabelEncoder()
    label_encoder.fit(dataset['Species'])
    dataset_lables = label_encoder.transform(dataset['Species'])

    dataset_features = torch.tensor(dataset_features.values)
    dataset_lables = torch.tensor(dataset_lables)

    separation = int(dataset.shape[0] * train_split)
    dataset_features_train = dataset_features[:separation]
    dataset_lables_train = dataset_lables[:separation]

    dataset_features_test = dataset_features[separation:]
    dataset_lables_test =  dataset_lables[separation:]

    print('train features dataset: ', dataset_features_train.shape)
    print('test features dataset: ', dataset_features_test.shape)

    return dataset_features_train, dataset_lables_train, dataset_features_test, dataset_lables_test
    
def dataset_loader(filepath='', batch_size=200):
    global dataset_iris_train
    if dataset_iris_train is None:
        print('dataset_iris is None, csv will be loaded...')
        dataset_features_train, dataset_lables_train, dataset_features_test, dataset_lables_test = dataset_parser(filepath=filepath)
        dataset_iris_train = IrisDataSet(dataset_features_train, dataset_lables_train)
        dataset_iris_test = IrisDataSet(dataset_features_test, dataset_lables_test)

    
    dataset_train = torch.utils.data.DataLoader(dataset_iris_train, batch_size=batch_size, shuffle=False)
    dataset_test = torch.utils.data.DataLoader(dataset_iris_test, batch_size=batch_size, shuffle=False)

    return dataset_train, dataset_test

dataset_train, dataset_test = dataset_loader('pytorch/transformer_explain/pytorch_nn_iris/data/iris.csv', batch_size=10)

print('Hello, World!')