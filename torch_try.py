import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
if_use_gpu = 0

training_data = torchvision.datasets.MNIST( root='./mnist/', train=True,  transform=torchvision.transforms.ToTensor(),  download=DOWNLOAD_MNIST, )

train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST( root='./mnist/',  train=False,  transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST, )

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
test_y = test_data.test_labels

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2)  )
        self.conv2 = nn.Sequential( nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2)  )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
if if_use_gpu:
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    start = time.time()
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch:', epoch, '|Step:', step, '|train loss:%.4f' % loss.data[0])

    print(epoch)


    duration = time.time() - start
    print('Training duation: %.4f'%duration)

cnn = cnn.cpu()
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == test_y) / test_y.size(0)
print('Test Acc: %.4f'%accuracy)


