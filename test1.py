import torch
from PIL import Image
from torchvision import transforms
from net import MyAlexNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

ROOT_test = ("./data/test")

model = MyAlexNet()
'''这里我用了个异常处理结构'''
try:
    model.load_state_dict(torch.load('./save_model/best_model.pth'))
except :
    '''如果模型是导入GPU训练的你需要把模型再导入回cpu'''
    model.load_state_dict(torch.load('./save_model/best_model.pth',map_location=torch.device('cpu')))

transform = transforms.Compose([
    transforms.Resize((224,224)),#统一大小
    transforms.ToTensor(),#转换数据
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])#归一化
])
Val_dataset = ImageFolder(ROOT_test,transform=transform)
Val_dataloder = DataLoader(Val_dataset)

#取文件路径
all_files = []
file_list = os.walk(ROOT_test)  # 获取当前路径下的所有文件和目录
for dirpath, dirnames, filenames in file_list:  # 从file_list中获得三个元素
    for file in filenames:
        all_files.append(os.path.join(dirpath, file))  # 用os.path.join链接文件名和路径，跟新进all_files列表里


classes = [
    '猫', '狗']#这里的答案顺序要和训练文件名字顺序一致
def prediect():
    model.eval()
    epoch=0
    for x,y in Val_dataloder:
        with torch.no_grad():
            outputs = model(x)
            outputs = torch.softmax(outputs,dim=1)#用softmax的特性在对数据进行一个标准化的概率预测值
            max_pred, predicted = torch.max(outputs, 1)#取出数据最大值和最大值的标签
            epoch+=1
            print(f'epoch：{epoch}  this picture maybe :{classes[predicted[0]]},pred：{max_pred.item()*100:.3f}%')


if __name__ == '__main__':
    prediect()
