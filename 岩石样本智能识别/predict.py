import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import Counter
from transforms import tta_test_transform
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def predict1(model1,model2,model3):
    # 读入模型
    model1 = load_checkpoint(model1)
    model2 = load_checkpoint(model2)
    model3 = load_checkpoint(model3)
    print('..... Finished loading model! ......')
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(int(os.path.basename(img_path).split('.')[0]))
        img1 = Image.open(img_path).convert('RGB')
        blending_y_pred=[]
        # print(type(img))
        ##*****************************model1********************************
        pred1 = []
        for i in range(8):
            img = tta_test_transform(size=300)(img1).unsqueeze(0)
            with torch.no_grad():
                out1 = model1(img)
            prediction1 = torch.argmax(out1, dim=1).cpu().item()
            pred1.append(prediction1)
        res1 = Counter(pred1).most_common(1)[0][0]
        blending_y_pred.append(res1)
        ##*****************************model2********************************
        pred2 = []
        for i in range(8):
            img = tta_test_transform(size=300)(img1).unsqueeze(0)
            with torch.no_grad():
                out2 = model2(img)
            prediction2 = torch.argmax(out2, dim=1).cpu().item()
            pred2.append(prediction2)
        res2 = Counter(pred2).most_common(1)[0][0]
        blending_y_pred.append(res2)
        ##*****************************model3********************************
        pred3 = []
        for i in range(8):
            img = tta_test_transform(size=300)(img1).unsqueeze(0)
            with torch.no_grad():
                out3 = model3(img)

            prediction3 = torch.argmax(out3, dim=1).cpu().item()
            pred3.append(prediction3)
        res3 = Counter(pred3).most_common(1)[0][0]
        blending_y_pred.append(res3)

        blending_res=Counter(blending_y_pred).most_common(1)[0][0]
        pred_list.append(blending_res)
    return _id, pred_list
def predict2(model1,model2,model3):
    # 读入模型
    model1 = load_checkpoint(model1)
    model2 = load_checkpoint(model2)
    model3 = load_checkpoint(model3)
    print('..... Finished loading model! ......')
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(int(os.path.basename(img_path).split('.')[0]))
        img1 = Image.open(img_path).convert('RGB')
        # print(type(img))
        ##*****************************model1********************************
        pred = []
        for i in range(8):
            img = tta_test_transform(size=300)(img1).unsqueeze(0)
            with torch.no_grad():
                out1 = model1(img)
                out2 = model2(img)
                out3 = model3(img)
                blending_y_pred=out1*0.4+out2*0.3+out3*0.4
            prediction = torch.argmax(blending_y_pred, dim=1).cpu().item()
            pred.append(prediction)
        res = Counter(pred).most_common(1)[0][0]
        pred_list.append(res)
    return _id, pred_list
if __name__ == "__main__":
    trained_model = ''
    model_name = ''
    val_fath=''
    bc_fath=''
    with open(val_fath,  'r')as f:
        imgs = f.readlines()

    # _id, pred_list = tta_predict(trained_model)
    _id, pred_list = predict1(trained_model)

    submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    submission.to_csv(bc_fath+ '{}_submission.csv'
                      .format(model_name), index=False, header=False)





