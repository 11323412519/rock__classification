import cv2
from PIL.Image import open as imread
import random

import numpy as np
def random_crop(image):
    print(image.shape)
    min_ratio = 0.6
    max_ratio = 1

    w, h = image.shape[0],image.shape[1]
    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    #image = image.crop((x, y, x + new_w, y + new_h))
    image=image[x:x + new_w,y:y + new_h]
    print(image.shape)


    cv2.imshow("block image", image)
    cv2.waitKey(0)




if __name__ == '__main__':
    img=cv2.imread('C:/Users/13234/Pictures/Saved Pictures/20.jpg')

    random_crop(img)

