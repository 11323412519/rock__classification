import torch.distributions
from torch.utils.data import DataLoader
from dataloader.data_toch import dataset_loader
from transforms import get_train_transform,get_test_transform,get_val_transform,tta_test_transform
from model.resnet import LeNet
from optim import SGD
from loss import CrossEntropyLoss
from  scheduler import *
from  chart.diagram import *
from  chart.ConfusionMatrix import *
import ttach as tta
import os
from ast import literal_eval
from tensorboardX import SummaryWriter
from torchsummary import summary
import argparse
from torch.utils.data.distributed import DistributedSampler
SumWriter=SummaryWriter(log_dir="../pytoch模板/log") #tensorboardX log路径
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
dataset='C:/Users/13234/Desktop/pytoch模板/dataset'
weight_path='C:/Users/13234/Desktop/pytoch模板/weight'

#***************************************************************************
#argparse命令输入参数获取
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", help="local device id on current node", type=int)
args = parser.parse_args()
#初始化进程组 并设置当前进程
n_gpus = 2
torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
torch.cuda.set_device(args.local_rank)
#**************************************************************************#
log_dir = 'C:/Users/13234/Desktop/pytoch模板/runs/exp1.0/lenet_epoch_10.pth'
lable = {'0': '类别一', '1': '类别二'}
num_class=len(lable.keys())
epochs=50
lr=0.001
batch_size=16
model_name='lenet'
#**************************************************************************#


#Data Augmentation
train_data,test_data,val_data=dataset_loader(dataset,get_train_transform(),get_test_transform(),get_val_transform(),split=True)
#Tensor类型转换
train_sampler=DistributedSampler(train_data)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,sampler=train_sampler)

test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
#模型加载
model=LeNet()
model=torch.nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),device_ids=[args.local_rank])
init_img=torch.zeros((1,3,224,224))
SumWriter.add_graph(model,init_img)
summary(model, input_size=(3, 224, 224))


##定义优化器与损失函数
optimizer=SGD(model,lr=lr)
criterion=CrossEntropyLoss()

#学习率调整
scheduler = CosineannealingLR(optimizer)



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
        x, y = x.cuda(args.local_rank), y.cuda(args.local_rank)
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
            X, y = X.cuda(args.local_rank), y.cuda(args.local_rank)
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
            X, y = X.cuda(args.local_rank), y.cuda(args.local_rank)
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
        train_sampler.set_epoch(epoch)
        train(model, train_loader,epoch)
        SumWriter.add_scalar("learning_rate",optimizer.param_groups[0]["lr"],epoch)
        scheduler.step()
        test(model, test_loader)

        # **************************模型的保存*****************************
        if epoch % 10 == 0 and epoch > 0 and args.local_rank== 0:
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
