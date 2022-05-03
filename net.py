import torch
import torch.nn.functional as f

'''Alex net '''
class MyAlexNet(torch.nn.Module):
    def __init__(self):
        '''定义子类和参数'''
        super(MyAlexNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,96,kernel_size=11,stride=4)
        self.mp = torch.nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = torch.nn.Conv2d(96,256,kernel_size=5,padding=2)
        self.conv3 = torch.nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 = torch.nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.conv5 = torch.nn.Conv2d(384,256,kernel_size=3)
        self.Linear1 = torch.nn.Linear(4096,4096)
        self.Linear2 = torch.nn.Linear(4096,2048)
        self.Linear3 = torch.nn.Linear(2048,1000)
        self.Linear4 = torch.nn.Linear(1000,2)

    def forward(self,x):
        '''网络构建'''
        batch_size =x.size(0)#x.Size([1, 3, 224, 224])
        x = f.relu(self.mp(self.conv1(x)))
        x = f.relu(self.mp(self.conv2(x)))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = self.mp(x)
        x = x.view(batch_size,-1)#（数据维度，数据长度）-1是指让电脑自动运算数据得到
        x = self.Linear1(x)
        x = f.dropout(x, p=0.5)  # 防止过拟合，有50%的数据随机失效
        x = self.Linear2(x)
        x = f.dropout(x,p=0.5)
        x = self.Linear3(x)
        x = f.dropout(x, p=0.5)
        x = self.Linear4(x)
        return x

x = torch.rand(1,3,224,224)
model = MyAlexNet()


