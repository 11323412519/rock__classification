import random
import torchvision
from PIL import Image, ImageFilter
from torchvision import transforms
class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img

class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def get_train_transform(mean=mean, std=std, size=0):
    train_transform = transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        # 随机裁剪 size 224*224
        torchvision.transforms.RandomCrop(224),
        # 中心裁剪 size 224*224
        torchvision.transforms.CenterCrop(224),
        # 将图片的尺寸 Resize 到128*128 不裁剪
        torchvision.transforms.Resize((224, 224)),
        # 转为张量并归一化到[0,1]（是将数据除以255），且会把H*W*C会变成C *H *W
        torchvision.transforms.ToTensor(),
        # 数据归一化处理，3个通道中的数据整理理到[-1, 1]区间。3个通道，故有3个值。该[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
        # ToTensor（）的[0，1]只是范围改变了， 并没有改变分布，mean和std处理后可以让数据正态分布
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
    return train_transform


def get_test_transform():
    return transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        # 随机裁剪 size 224*224
        torchvision.transforms.RandomCrop(224),
        # 中心裁剪 size 224*224
        torchvision.transforms.CenterCrop(224),
        # 将图片的尺寸 Resize 到128*128 不裁剪
        torchvision.transforms.Resize((224, 224)),
        # 转为张量并归一化到[0,1]（是将数据除以255），且会把H*W*C会变成C *H *W
        torchvision.transforms.ToTensor(),
        # 数据归一化处理，3个通道中的数据整理理到[-1, 1]区间。3个通道，故有3个值。该[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
        # ToTensor（）的[0，1]只是范围改变了， 并没有改变分布，mean和std处理后可以让数据正态分布
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def get_val_transform():
    return transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # 转为张量并归一化到[0,1]（是将数据除以255），且会把H*W*C会变成C *H *W
        torchvision.transforms.ToTensor(),
        # 数据归一化处理，3个通道中的数据整理理到[-1, 1]区间。3个通道，故有3个值。该[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
        # ToTensor（）的[0，1]只是范围改变了， 并没有改变分布，mean和std处理后可以让数据正态分布
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def tta_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # transforms.CenterCrop(size),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        RandomRotate(15, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])






