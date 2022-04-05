from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
def model_load(out_features=4):
    model = EfficientNet.from_name('efficientnet-b4')
    state_dict = torch.load('efficientnet-b4.pth')#加载预训练模型
    model.load_state_dict(state_dict)#获取全连接的in_features
    in_fea = model._fc.in_features #读取全连接层的in_features
    #要改的是最终输出的特征维度out_features(假设分类数为40)
    model._fc = nn.Linear(in_features=in_fea, out_features= out_features,bias=True)
    return model
