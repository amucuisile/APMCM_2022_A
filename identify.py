import cv2
import easyocr
import ssl
import os
import pandas as pd
from tqdm import tqdm
# 避免 ssl 证书报错导致的下载模型出错的问题
ssl._create_default_https_context = ssl._create_unverified_context
# 初始化 easyocr
reader = easyocr.Reader(['en'])
# thresholding 二值化处理，提高识别精准度
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# 批量读取图片并进行预处理
def readImages(dirpath):
    files = os.listdir(dirpath)
    result = {}
    for file in tqdm(files):
        # 判断是否是图片
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
            img = cv2.imread(dirpath + '/' + file, cv2.IMREAD_GRAYSCALE)
            img_v1 = thresholding(img[19:37, 50:115])
            img_v2 = thresholding(img[37:57, 50:115])
        name = os.path.split(file)[1].split('.')[0]
        result[int(name)] = [img_v1, img_v2]
    return result
# 识别图片中的文字
def recognize_text(image):
    try:
        text = reader.readtext(image, detail=0, add_margin=1)
        return ''.join(filter(str.isdigit, text[0]))[0:4]
    except:
        # 如果识别失败，返回空字符串，待后续处理
        return ''
if __name__ == "__main__":
    # 读取附件
    df = pd.read_excel(os.path.abspath("Attachment 2.xlsx"))
    print('图片读取中...')
    dirpath = 'images'  # 图片所在的文件夹
    images = readImages(dirpath)
    print('图片读取完成')
    print('文字识别中...')
    # 排序
    images = dict(sorted(images.items(), key=lambda x: x[0]))
    index = 0
    for key, value in tqdm(images.items()):
        df.loc[index, 'Time'] = key
        df.loc[index, '1# Temperature'] = recognize_text(value[0])
        df.loc[index, '2# Temperature'] = recognize_text(value[1])
        index+=1
    print('识别完成')
    # 保存结果
    df.to_excel('result.xlsx', index=False)