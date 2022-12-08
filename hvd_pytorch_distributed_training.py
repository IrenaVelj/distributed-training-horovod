import torch
import horovod.torch as hvd
import torchvision
from torch import optim
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward (self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    print(hvd.size())
    print(hvd.local_rank())

# Define dataset...
train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = torchvision.transforms.ToTensor() ,download = True)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                                num_replicas=hvd.size(), 
                                                                rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

# Build model...
model = NeuralNet()
model.cuda()

optimizer = optim.SGD(model.parameters(), 0.001*hvd.size())

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.reshape(-1, 28*28).cuda()
        target = target.cuda()
        
        output = model(data)
        loss = nn.functional.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, batch_idx * len(data), len(train_sampler), loss.item()))