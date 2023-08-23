import torch
import torch.utils.data as Data

# 设定随机种子
torch.manual_seed(1)

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(0.5, 5, 10)

# 将数据转换为torch的dataset格式
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 将torch_dataset置入Dataloader中
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE, # 批大小
    # 若dataset中的样本不能被batch_size整除的话，最后剩余多少就使用多少
    shuffle=True,  # 是否随机打乱顺序
    num_workers=2, # 多线程读取数据的线程数
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch:', epoch, '|Step:', step, '|batch_x:',
              batch_x.numpy(), '|batch_y', batch_y.numpy())
        
