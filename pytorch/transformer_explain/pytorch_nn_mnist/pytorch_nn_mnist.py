import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 数据加载定义
def data_loader(batch_size=200):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

# 网络结构定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, -1)
    
# 模型训练定义
net = Net()
print(net)

# 超参数定义
learning_rate = 1e-2
log_interval = 10
batch_size = 200
epoch_num = 10
momentum = 0.9

# 创建一个优化器
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# 创建损失函数
criterion = nn.NLLLoss()

# 模型训练
for epoch in range(epoch_num):
    train_loader, _ = data_loader(batch_size=batch_size)
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据格式转换：(batch_size, 1, 28, 28) to (batch_size, 28 * 28)
        data = data.view(-1, 28 * 28)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

# 模型测试
test_loss = 0
correct = 0
_, test_loader = data_loader()
for data, target in test_loader:
    data = data.view(-1, 28 * 28)
    net_out = net(data)

    # sum up batch loss
    test_loss += criterion(net_out, target).data
    # 获得最大log_probability的索引
    pred = net_out.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), 
    100. * correct / len(test_loader.dataset)))