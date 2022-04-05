import torch
from transforms import get_val_transform
from dataloader.data_toch import val_loader
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from chart.ConfusionMatrix import *
import ttach as tta
import torch.nn as nn
from torchvision import models
from model.resnet import LeNet,alexnet
def load_checkpoint(filepath):
    model_ft = models.resnet18()  # 调用pytorch已经封装好的模型resnet18，并且自动预训练。
    num_ftrs = model_ft.fc.in_features  # 获取resnet18的fc（函数，这里用的nn.Linear）层的输入的特征
    model_ft.fc = nn.Linear(num_ftrs, 4)  # 将输出维度改为2，因为我们这里要做2分类

    model_CKPT = torch.load(filepath)
    model_ft.load_state_dict(model_CKPT['model'])
    print('loading checkpoint!')

    return model_ft
def load_checkpoint1(filepath):
    model=LeNet()
    model_CKPT = torch.load(filepath)
    model.load_state_dict(model_CKPT['model'])
    print('loading checkpoint!')

    return model
def load_checkpoint2(filepath):
    model=alexnet()

    model_CKPT = torch.load(filepath)
    model.load_state_dict(model_CKPT['model'])
    print('loading checkpoint!')

    return model
def predict1(model1,model2,model3,num_class):
    # 读入模型
    model1 = load_checkpoint(model1)
    model2 = load_checkpoint1(model2)
    model3 = load_checkpoint2(model3)
    tta_model1 = tta.ClassificationTTAWrapper(model1, tta.aliases.five_crop_transform(224,224))
    tta_model2 = tta.ClassificationTTAWrapper(model2, tta.aliases.five_crop_transform(224,224))
    tta_model3 = tta.ClassificationTTAWrapper(model3, tta.aliases.five_crop_transform(224,224))
    model1.eval(),model2.eval(),model3.eval()
    label = [label for _, label in lable.items()]
    confusion = ConfusionMatrix(num_classes=num_class, labels=label)
    val_size = len(val_data.dataset)
    correct_sample = 0
    with torch.no_grad():
        for X, y in val_data:
            X = X.to(device)  # 将double数据转换为float
            y = y.to(device)
            out1 = tta_model1(X)
            out2 = tta_model2(X)
            out3 = tta_model3(X)
            out=out1*weight1+out2*weight2+out3*weight3
            correct_sample += (out.argmax(1) == y).type(torch.float).sum().item()
            ret, predictions = torch.max(out.data, 1)
            confusion.update(predictions.numpy(), y.numpy())
        print(f"模型最终的准确率为:{(correct_sample / val_size) * 100}%")
        confusion.plot(name="tta_confusionmatrix.png")
        confusion.summary()


def predict2(model1, model2, model3):
    # 读入模型
    model1 = load_checkpoint(model1)
    model2 = load_checkpoint(model2)
    model3 = load_checkpoint(model3)
    tta_model1 = tta.ClassificationTTAWrapper(model1, tta.aliases.five_crop_transform(224,224))
    tta_model2 = tta.ClassificationTTAWrapper(model2, tta.aliases.five_crop_transform(224,224))
    tta_model3 = tta.ClassificationTTAWrapper(model3, tta.aliases.five_crop_transform(224,224))
    val_preds = []
    val_trues = []
    model1.eval(), model2.eval(), model3.eval()
    with torch.no_grad():
        for x, y in val_data:
            blending_y=[]
            x = x.to(device)  # 将double数据转换为float
            y = y.to(device)
            out1 = tta_model1(x)
            out2 = tta_model2(x)
            out3 = tta_model3(x)
            blending_y.append(out1.argmax(dim=1))
            blending_y.append(out2.argmax(dim=1))
            blending_y.append(out3.argmax(dim=1))
            res = Counter(blending_y).most_common(1)[0][0]
            val_preds.extend(res.detach().cpu().numpy())
            val_trues.extend(y.detach().cpu().numpy())
        sklearn_accuracy = accuracy_score(val_preds, val_trues)
        sklearn_precision = precision_score(val_trues, val_preds, average='micro')
        sklearn_recall = recall_score(val_trues, val_preds, average='micro')
        sklearn_f1 = f1_score(val_trues, val_preds, average='micro')
        print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy,
                                                                                                  sklearn_precision,
                                                                                                  sklearn_recall,
                                                                                                  sklearn_f1))
if __name__ == "__main__":
    lable = {'0': 'cloudy', '1': 'rain','2':'shine','3':'sumrise'}
    dataset_path= 'dataset'
    weight1,weight2,weight3=0.3,0.3,0.4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_transforms=get_val_transform()
    val_data=val_loader(dataset_path,val_transforms)
    val_data = DataLoader(dataset=val_data, batch_size=16)
    trained_model1 = 'C:/Users/13234/Desktop/pytoch模板/runs/exp1.0 - resnet/lenet_epoch_10.pth'
    trained_model2 = 'C:/Users/13234/Desktop/pytoch模板/runs/exp5.0 - lenet/lenet_epoch_50.pth'
    trained_model3 = 'C:/Users/13234/Desktop/pytoch模板/runs/exp5.0-alexnet/lenet_epoch_50.pth'
    predict1(trained_model1,trained_model2,trained_model3,num_class=len(lable.keys()))

