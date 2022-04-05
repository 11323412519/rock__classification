import numpy as np
import matplotlib.pyplot as plt
import cv2


def display_blocks(divide_image):#
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j,:])
            plt.axis('off')
            plt.title('block:'+str(i*n+j+1))

def create_image_block( image, row_number, col_number):
    block_row = np.array_split(image, row_number, axis=0)  # 垂直方向切割，得到很多横向长条
    print(image.shape)
    img_blocks = []
    for block in block_row:
        block_col = np.array_split(block, col_number, axis=1)  # 水平方向切割，得到很多图像块
        img_blocks += [block_col]

    for i in range(0,2):
        for j in range(0,2):

            cv2.imshow("block image", img_blocks[i][j])  # 第3行第2列图像块
            cv2.waitKey(0)
            cv2.imwrite('C:/Users/13234/Desktop/BDCI2020-seg/scr/img'+str(i)+'-'+str(j)+'.jpg', img_blocks[i][j])  # 保存


img = cv2.imread('C:/Users/13234/Pictures/Saved Pictures/20.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

divide_image1=create_image_block(img,2,2)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数


