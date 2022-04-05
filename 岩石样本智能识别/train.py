from torch.utils.data import DataLoader
from dataloader.data_toch import dataset_loader
from transforms import get_train_transform,get_test_transform,get_val_transform
from model.resnet import alexnet
import torchvision.models as models
import torch.nn as nn
from optim import SGD
from loss import CrossEntropyLoss
from  scheduler import *
from  chart.diagram import *
from  chart.ConfusionMatrix import *
import ttach as tta
import os
from  model import  resnet
from ast import literal_eval
from tensorboardX import SummaryWriter
SumWriter=SummaryWriter(log_dir="../pytoch模板/log")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset='../pytoch模板/dataset'
weight_path='../pytoch模板/weight'
from torchsummary import summary #网络结构打印


#**************************************************************************#
log_dir = '../pytoch模板/runs/exp3.0/lenet_epoch_30.pth'
lable = {'深灰色泥岩': 0, '黑色煤': 1, '灰色细砂岩': 2, '浅灰色细砂岩': 3, '深灰色粉砂质泥岩': 4, '灰黑色泥岩': 5, '灰色泥质粉砂岩': 6}
num_class=len(lable.keys())
epochs=50
lr=0.001
batch_size=16
model_name='lenet'
#**************************************************************************#


#Data Augmentation
train_data,test_data,val_data=dataset_loader(dataset,get_train_transform(),get_test_transform(),get_val_transform(),split=True)
#Tensor类型转换
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
#模型加载

model=models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
channel_in = model.fc.in_features
model.fc = nn.Linear(channel_in,7)

init_img=torch.zeros((1,3,224,224),device=device)
SumWriter.add_graph(model,init_img)
summary(model, input_size=(3, 224, 224))


##定义优化器与损失函数
optimizer=SGD(model,lr=lr)
criterion=CrossEntropyLoss()

#学习率调整
scheduler = ExponentialLR(optimizer)



train_loss = []
test_loss = []
train_acc=[]
test_acc=[]

#traing_model
def train(model, train_loader,epoch):
    model.train()
    train_size = len(train_loader.dataset)
    running_loss, running_correct = 0.0, 0.0
    # **************************************训练*********************************
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += (output.argmax(1) == y).type(torch.float).sum().item()

    running_loss /= train_size
    running_correct /= train_size
    train_loss.append(running_loss)
    train_acc.append(100 * running_correct)
    SumWriter.add_scalar("train_loss",running_loss,global_step=epoch)
    print(f"Epoch:{epoch}, train acc:{(100 * running_correct):>0.1f}% train loss:{(running_loss):>4f}",end='')

# test_model
def test(model, test_loader):
    model.eval()
    test_size = len(test_loader.dataset)
    running_loss, running_correct = 0.0, 0.0
    # **************************************测试*********************************
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            running_loss += criterion(pred, y).item()
            running_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    running_loss /= test_size
    running_correct /= test_size
    test_loss.append(running_loss)
    test_acc.append(100 * running_correct)
    SumWriter.add_scalar("accuracy", 100 * running_correct, global_step=epoch)
    print(f" test acc:{(100 * running_correct):>0.1f}% test loss:{running_loss:>4f} \n")
#val_model
def val():
    # Precision 准确率| Recall召回率 | Specificity 特效度
    # 混淆矩阵
    label = [label for _, label in lable.items()]
    confusion = ConfusionMatrix(num_classes=num_class, labels=label)
    tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(224,224))
    val_size = len(val_loader.dataset)
    correct_sample=0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = tta_model(X)
            correct_sample += (pred.argmax(1) == y).type(torch.float).sum().item()
            ret, predictions = torch.max(pred.data, 1)
            confusion.update(predictions.numpy(), y.numpy())
        confusion.plot()
        confusion.summary()
    print(f"模型最终的准确率为:{(correct_sample/val_size)*100}%")

if __name__ == '__main__':
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
        with open('runs/{}/c.txt'.format('exp' + str(start_epoch/10)), "r") as f:
            c_list = list(f.readlines())
            train_acc,test_acc,train_loss,test_loss=literal_eval(c_list[0]),literal_eval(c_list[1]),literal_eval(c_list[2]),literal_eval(c_list[3])
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
    for epoch in range(start_epoch+1,epochs+1):
        train(model, train_loader,epoch)
        SumWriter.add_scalar("learning_rate",optimizer.param_groups[0]["lr"],epoch)
        scheduler.step()
        test(model, test_loader)

        # **************************模型的保存*****************************
        if epoch % 10 == 0 and epoch > 0:
            isExists = os.path.exists('runs/{}'.format('exp'+str(epoch/10)))
            # 判断结果
            if not isExists:
                os.makedirs('runs/{}'.format('exp'+str(epoch/10)))
            save_model_pth = 'runs/{}/'.format('exp' + str(epoch / 10))

            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, os.path.join(save_model_pth, '{}_epoch_{}.pth'.format(model_name,epoch)))
            with open('runs/{}/c.txt'.format('exp' + str(epoch / 10)), "w") as f:
                for i in train_acc,test_acc,train_loss,test_loss:
                    f.writelines(str(i) + "\n")
            acc_chart(train_acc, test_acc, epoch,path=save_model_pth)
            loss_chart(train_loss, test_loss, epoch,path=save_model_pth)
    #Precision 准确率| Recall召回率 | Specificity 特效度
    val()















