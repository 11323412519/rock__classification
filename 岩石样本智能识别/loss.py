import torch.nn as nn
import torch.nn.functional as F
class LabelSmoothingCrossEntropys(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

def CrossEntropyLoss():
    return nn.CrossEntropyLoss()
def BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()
#def LabelSmoothSoftmaxCE():
    #return LabelSmoothSoftmaxCE()
def LabelSmoothingCrossEntropy():#标签平滑
    return LabelSmoothingCrossEntropys()




