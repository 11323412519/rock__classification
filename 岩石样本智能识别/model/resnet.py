import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(44944,120)
        self.fc2=nn.Linear(120,10)
        self.fc3=nn.Linear(10,4)
    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        #这句话一般出现在model类的forward函数中，具体位置一般都是在调用分类器之前。
        #分类器是一个简单的nn.Linear()结构，输入输出都是维度为一的值，x = x.view(x.size(0), -1)
        #这句话的出现就是为了将前面多维度的tensor展平成一维
        #x = x.view(batchsize, -1)中batchsize指转换后有几行，
        #而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        out=out.view(out.size(0),-1)
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
        return out


class alexnet(nn.Module):

    def __init__(self, num_classes=4):
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out





