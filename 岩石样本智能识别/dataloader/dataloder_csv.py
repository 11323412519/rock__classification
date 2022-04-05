import csv
import shutil
import os
#所有的图像必须保存在同一文件夹下  .csv文件中有2列 第一列是图像名，第二列为类名 如果 123.jpg A.
#需修改 original_path(初始图像数据集路径)和csv_path(csv路径) 分好的类文件夹会保存在dataset文件夹下

target_path = '../dataset/'
original_path = 'C:/Users/13234/Desktop/新建文件夹/'
csv_path='C:/Users/13234/Desktop/新建文件夹/indexData.csv'
with open(csv_path,"rt", encoding=" UTF-8-sig") as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    for row in rows:
        if os.path.exists(target_path+row[1]) :
            full_path = original_path + row[0]
            shutil.move(full_path,target_path + row[1] +'/')
        else :
            os.makedirs(target_path+row[1])
            full_path = original_path + row[0]
            shutil.move(full_path,target_path + row[1] +'/')

