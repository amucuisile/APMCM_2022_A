import os 
import cv2
import numpy as np

def CropImage4File(filepath):
    pathDir =  os.listdir(filepath)    # 列出文件路径中的所有路径或文件
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isfile(child):
            img = cv2.imread(child,flags=1)
            b = np.mean(img[:,:,0])
            g = np.mean(img[:,:,1])
            r = np.mean(img[:,:,2])
            print(r)
            
           
if __name__ == "__main__":
    filepath = 'images'  # 图片所在的文件夹
    CropImage4File(filepath)



