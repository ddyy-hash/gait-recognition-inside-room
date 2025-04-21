# made by Liang Ren
import math
import os
import cv2
import numpy as np

T_H = 64
T_W = 64

def cut_img(img):
    # 剪影中白色像素过少可能无法有效识别。
    if img.sum() <= 10000:
        return None, True
    # 得到最高点和最低点
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # 由于人的身高大于宽度，因此使用身高来计算缩放比例。
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # 获取x轴的中值，并将其视为人的x中心。
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        return None, True
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8'), False


def cut(image):
    '''
    通过找到人的最小最大高度与宽度把人的轮廓分割出来，、
    因为原始轮廓图为二值图，因此头顶为将二值图像列相加后，形成一列后第一个像素值不为0的索引。
    同理脚底为形成一列后最后一个像素值不为0的索引。
    人的宽度也同理。
    :param image: 需要裁剪的图片 N*M的矩阵
    :return: temp:裁剪后的图片 size*size的矩阵。flag：是否是符合要求的图片
    '''
    image = np.array(image)
    # 找到人的最小最大高度与宽度
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    # 设置切割后图片的大小，为size*size，因为人的高一般都会大于宽
    size = height_max - height_min
    temp = np.zeros((size, size))
    # 将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    # l = (width_max-width_min)//2
    # r = width_max-width_min-l
    # 以头为中心，将将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    l1 = head_top - width_min
    r1 = width_max - head_top
    # 若宽大于高，或头的左侧或右侧身子比要生成图片的一般要大。则此图片为不符合要求的图片
    flag = False
    if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
        flag = True
        return temp, flag
    # centroid = np.array([(width_max+width_min)/2,(height_max+height_min)/2],dtype='int')
    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]
    temp = cv2.resize(temp, (64, 64), interpolation=cv2.INTER_AREA)
    return temp, flag

#加载背景视频和行人视频
def load_video(video_path, background_path):
    # 打开背景视频文件
    bg_cap = cv2.VideoCapture(background_path)

    # 获取背景图（选择第一帧作为背景图）
    ret, background_frame = bg_cap.read()
    if not ret:
        print("无法读取背景视频的第一帧")
        exit()
    bg_cap.release()

    # 打开行人视频文件
    pedestrian_cap = cv2.VideoCapture(video_path)
    if not pedestrian_cap.isOpened():
        print("无法打开行人视频")
        exit()
    frame_count = 0
    person_video = []
    silo = []
    while True:
        ret, pedestrian_frame = pedestrian_cap.read()

        if not ret:
            break  # 视频结束
        person_video.append(pedestrian_frame)
        # 背景减除，得到行人轮廓
        pedestrian_contour = cv2.absdiff(pedestrian_frame, background_frame)

        # 转换为灰度图像并进行阈值化处理，膨胀腐蚀操作，提取轮廓
        gray = cv2.cvtColor(pedestrian_contour, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        thresholded = cv2.erode(thresholded, np.ones((3, 3), np.uint8), iterations=1)
        thresholded = cv2.dilate(thresholded, np.ones((2, 2), np.uint8), iterations=1)
        #两种裁剪方法，将行人轮廓图裁剪成模型需要的大小
        thresholded, flag = cut_img(thresholded)
        # thresholded, flag = cut(thresholded)

        if flag is True:
            continue
        thresholded = thresholded[:, 10: -10].astype('float32') / 255.0
        silo.append(thresholded)
        frame_count += 1
        #cv2.imshow('Pedestrian Contour', thresholded)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频捕捉对象
    pedestrian_cap.release()
    #cv2.destroyAllWindows()
    begin = int(0.1*frame_count)
    end = math.ceil(0.1*frame_count)
    '''l = silo[begin:-end]
    print(len(l))
    for i in l:
        cv2.imshow('Pedestrian Contour', i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()'''
    return person_video, silo[begin:-end]


#主函数，通过加载的行人视频文件路径得到背景视频路径，加载这两个视频
def load(video_path):
    video_name = video_path.split('\\')[-1]
    split_name = video_name.split('-')
    background_name = split_name[0] + '-bkgrd-' + split_name[-1]
    directory_path = os.path.dirname(video_path)
    background_path = os.path.join(directory_path, background_name)
    return load_video(video_path, background_path)


if __name__ == '__main__':
    load('D:\\GaitSet\\Discriminator\\video\\116-nm-01-090.avi')
