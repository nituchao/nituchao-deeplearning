import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
	assert 0 <= dropout <= 1
	# 在本情况中，所有元素都被丢弃
	if dropout == 1:
		return torch.zeros_like(X)
	# 在本情况中，所有元素都被保留
	if dropout == 0:
		return X
	mask = (torch.rand(X.shape) > dropout).float()
	return mask * X / (1.0 - dropout)

# 我们可以通过下面几个例子来测试dropout_layer函数。我们将输入X通过暂退法操作，暂退概率分别为0、0.5和1
# X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

# 使用Fashion-MNIST数据集
# 定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256


dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
	def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, is_training = True):
		super(Net, self).__init__()
		self.num_inputs = num_inputs
		self.training = is_training
		self.lin1 = nn.Linear(num_inputs, num_hiddens1)
		self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
		self.lin3 = nn.Linear(num_hiddens2, num_outputs)
		self.relu = nn.ReLU()

	def forward(self, X):
		H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
		# 只有在训练模型时才使用dropout
		if self.training == True:
			# 在第一个全连接层之后添加一个dropout层
			H1 = dropout_layer(H1, dropout1)
		H2 = self.relu(self.lin2(H1))
		if self.training == True:
			# 在第二全连接之后添加一个dropout层
			H2 = dropout_layer(H2, dropout2)
		out = self.lin3(H2)
		return out

net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)

# 训练模型
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)











