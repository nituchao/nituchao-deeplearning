import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing

##############################################################################
# 数据加载
##############################################################################
class IrisDataSet(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def dataset_parser(filepath=''):
    dataset = pd.read_csv(filepath, header=0, usecols=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
    dataset_features = dataset[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
    
    label_encoder=preprocessing.LabelEncoder()
    label_encoder.fit(dataset['Species'])
    dataset_lables = label_encoder.transform(dataset['Species'])

    dataset_features = torch.tensor(dataset_features.values)
    dataset_lables = torch.tensor(dataset_lables)

    return dataset_features, dataset_lables
    
def dataset_loader(filepath='', batch_size=200):
    dataset_features, dataset_lables = dataset_parser(filepath=filepath)

    dataset_iris = IrisDataSet(dataset_features, dataset_lables)
    dataset_train = torch.utils.data.DataLoader(dataset_iris, batch_size=batch_size, shuffle=True)
    dataset_test = torch.utils.data.DataLoader(dataset_iris, batch_size=batch_size, shuffle=True)

    return dataset_train, dataset_test

##############################################################################
# 网络定义
##############################################################################
class Net(nn.Module):
    def __init__(self) :
        super(Net, self).__init__()
        self.layer1 = nn.Linear(4, 128, bias=True).double()
        self.layer2 = nn.Linear(128, 128, bias=True).double()
        self.layer3 = nn.Linear(128, 10, bias=True).double()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x, dim=1)
    
net = Net()
print(net)

##############################################################################
# 模型训练
##############################################################################
learning_rate = 1e-3
batch_size =  10
log_interval = 10

# 创建一个随机梯度下降（stochastic gradient descent）优化器
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# 创建一个损失函数
criterion = nn.NLLLoss()

# 运行主训练循环
epoch_num = 1000
for epoch in range(epoch_num):
    # 数据加载
    dataset_train, _ = dataset_loader('./data/iris.csv', batch_size=batch_size)
    for batch_idx, (data, target) in enumerate(dataset_train):
        
        # 将数据大小变为(batch_size, 1 x 4)
        data = data.view(-1, 4)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()

        if batch_idx + 1 == batch_size:
            print('Train Epoch: {} [{}/{}] ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, epoch + 1,  epoch_num, 100. * epoch / epoch_num, loss))

# 模型评估
test_loss = 0
correct = 0
_, dataset_test = dataset_loader('./data/iris.csv', batch_size=batch_size)
for data, target in dataset_test:
    data = data.view(-1, 4)
    net_out = net(data)
    # sum up batch loss
    test_loss += criterion(net_out, target).data
    pred = net_out.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(dataset_test.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataset_test.dataset),
        100. * correct / len(dataset_test.dataset)))