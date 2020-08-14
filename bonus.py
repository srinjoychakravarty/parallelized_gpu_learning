import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck
import numpy as np
import timeit

num_classes = 1000

class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1).to('cuda:0')
        self.seq4 = nn.Sequential(self.layer2).to('cuda:3')
        self.seq3 = nn.Sequential(self.layer3).to('cuda:2')
        self.seq2 = nn.Sequential(self.layer4, self.avgpool).to('cuda:1')
        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq1(x).to('cuda:3')
        x = self.seq4(x).to('cuda:2')
        x = self.seq3(x).to('cuda:1')
        x = self.seq2(x)
        return self.fc(x.view(x.size(0), -1))
	
num_batches = 3
batch_size = 120
image_w = 128
image_h = 128

def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001)
    one_hot_indices = torch.LongTensor(batch_size).random_(0, num_classes).view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes).scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()		
		
num_repeat = 10

stmt = "train(model)"

setup = "import torchvision.models as models;" + "model = models.resnet50(num_classes = num_classes).to('cuda:0')"
single_gpu_runtime = timeit.repeat(stmt, setup, number = 1, repeat = num_repeat, globals = globals())
single_gpu_mean, single_gpu_std_dev = np.mean(single_gpu_runtime), np.std(single_gpu_runtime)

setup_2_gpu = "model = ModelParallelResNet50()"
dual_gpu_runtime = timeit.repeat(stmt, setup_2_gpu, number = 1, repeat = num_repeat, globals = globals())
dual_gpu_mean, dual_gpu_std_dev = np.mean(dual_gpu_runtime), np.std(dual_gpu_runtime)