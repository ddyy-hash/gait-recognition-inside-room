from ultralytics import YOLO
import cv2
import numpy as np
import math
class yolo_seg():
    model = None
    def __init__(self):
        # Load the model
        self.model = YOLO("./fea_data/seg_model.pt")  # load an official YOLO model

    def cut_img(self,img, T_W=64, T_H=64):
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
        #
        for i in range(sum_column.size):
            if sum_column[i] > sum_point / 2:
                x_center = i
                break
        #
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

    def calculate_center(self, contour_img):
        #从二值化轮廓图中计算人物中心坐标
        gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                print(cX)
                return cX  # 只需水平方向坐标
        return None

    def judge_direction(self, centers, threshold=3):
        # 计算前后帧的差值
        differences = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        # 忽略微小变化，只考虑显著位移
        significant_diffs = [d for d in differences if abs(d) > threshold]

        if not significant_diffs:  # 如果没有显著位移，使用所有数据
            significant_diffs = differences

        for d in significant_diffs:
            print(d)

        # 根据有显著差异的帧判断方向
        avg_movement = np.mean(significant_diffs)
        print("avg:\n")
        print(avg_movement)
        return avg_movement < 0  # 若中心持续左移，需要翻转

    def load_video(self,video_path):
        # 打开行人视频文件
        #更新行人方向检测，确保行人行进方向一致
        pedestrian_cap = cv2.VideoCapture(video_path)
        if not pedestrian_cap.isOpened():
            print("无法打开行人视频")
            exit()

        # 获取视频总帧数
        total_frames = int(pedestrian_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算10%位置的帧号
        start_frame = int(total_frames * 0.1)
        # 设置视频位置到10%处
        pedestrian_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # 预处理阶段：先读取前15帧来确定方向
        preview_frames = []
        preview_contours = []
        centers = []

        for _ in range(15):  # 读取前15帧
            ret, frame = pedestrian_cap.read()
            if not ret:
                break
            contour = self.get_single_mask_from_cvimage(frame)
            center = self.calculate_center(contour)
            if center is not None:
                centers.append(center)
            preview_frames.append(frame)
            preview_contours.append(contour)

        # 确定方向
        flip_required = False
        if len(centers) >= 5:
            flip_required = self.judge_direction(centers)

        # 重置视频到开始位置
        pedestrian_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = 0
        person_video = []
        silo = []
        while True:
            ret, pedestrian_frame = pedestrian_cap.read()
            # 视频结束
            if not ret:
                break

            '''#original video
            person_video.append(pedestrian_frame)'''

            #得到行人轮廓
            pedestrian_contour = self.get_single_mask_from_cvimage(pedestrian_frame)
            # cv2.imshow('Pedestrian Contour', pedestrian_contour)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # 根据确定的方向进行翻转
            if flip_required:
                pedestrian_frame = cv2.flip(pedestrian_frame, 1)  # 水平翻转
                pedestrian_contour = cv2.flip(pedestrian_contour, 1)


            # original video
            person_video.append(pedestrian_frame)

            # 转换为灰度图像并进行阈值化处理，膨胀腐蚀操作，提取轮廓
            gray = cv2.cvtColor(pedestrian_contour, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            thresholded = cv2.erode(thresholded, np.ones((3, 3), np.uint8), iterations=1)
            thresholded = cv2.dilate(thresholded, np.ones((2, 2), np.uint8), iterations=1)

            # 将行人轮廓图裁剪成模型需要的大小
            thresholded, flag = self.cut_img(thresholded)
            if flag is True:
                continue

            thresholded = thresholded[:, 10: -10].astype('float32') / 255.0
            silo.append(thresholded)
            frame_count += 1
            # print("\n {} \n".format(gait_dataset.npy))
            cv2.imshow('Pedestrian Contour', thresholded)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放视频捕捉对象
        pedestrian_cap.release()
        cv2.destroyAllWindows()
        begin = int(0.1 * frame_count)
        end = math.ceil(0.1 * frame_count)
        # l = silo[begin:-end]
        # print(len(l))
        # for i in l:
        #     cv2.imshow('Pedestrian Contour', i)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()
        return person_video, silo[begin:-end]

    def get_single_mask_from_cvimage(self, image, only_one = True):
        # Predict with the model
        # results = self.model(image)  # predict on an image
        results = self.model.predict(image)  # predict on an image
        # print("-",end="")
        # Create an empty mask for segmentation
        segmentation_mask = np.zeros_like(image, dtype=np.uint8)
        # only results[0] have meaning
        actual_result = results[0]
        if len(actual_result)<1:
            return segmentation_mask

        # print(actual_result.names)
        # Iterate through the detected masks
        for j, mask in enumerate(actual_result.masks.xy):
            # Convert the class tensor to an integer
            class_id = int(actual_result.boxes.cls[j].item())  # Extract the class ID as an integer

            # Check if the detected class corresponds to 'person' (class ID 0)
            if class_id == 0:
                # Convert mask coordinates to an integer format for drawing
                mask = np.array(mask, dtype=np.int32)

                # Fill the segmentation mask with color (e.g., white for people)
                cv2.fillPoly(segmentation_mask, [mask], (255, 255, 255))

            #only deal with the biggest
            if only_one:
                break

        return segmentation_mask

    # class Results(SimpleClass):
    #     这个类被设计用来存储和操作从YOLO模型得到的推理结果。它封装了处理检测、分割、姿态估计和分类任务结果的功能。以下是对这些属性和方法的解释：
    #     属性： - orig_img：原始图像的NumPy数组。 - orig_shape：原始图像的尺寸，格式为(height, width)。
    #     - boxes：包含检测到的边界框的对象。
    #     - masks：包含检测到的掩码（用于分割任务）的对象。
    #     - probs：包含分类任务的类别概率的对象。
    #     - keypoints：包含检测到的每个对象的关键点的对象。
    #     - obb：包含定向边界框的对象。
    #     - speed：包含预处理、推理和后处理速度的字典。
    #     - names：将类别ID映射到类别名称的字典。
    #     - path：图像文件的路径。
    #     - _keys：用于内部使用的属性名称元组。
    def get_mask_from_cvimage(self,image):
        # Predict with the model
        # results = self.model(image)  # predict on an image
        results = self.model.predict(image)  # predict on an image
        # Create an empty mask for segmentation
        segmentation_mask = np.zeros_like(image, dtype=np.uint8)
        # only results[0] have meaning
        actual_result = results[0]
        if len(actual_result)<1:
            return (None,None)

        print(actual_result.names)
        print(actual_result.boxes.cls)
        # Iterate through the detected masks
        counter=1
        for j, mask in enumerate(actual_result.masks.xy):
            # Convert the class tensor to an integer
            class_id = int(actual_result.boxes.cls[j].item())  # Extract the class ID as an integer

            # Check if the detected class corresponds to 'person' (class ID 0)
            if class_id == 0:
                # Convert mask coordinates to an integer format for drawing
                mask = np.array(mask, dtype=np.int32)

                # Fill the segmentation mask with color (e.g., white for people)
                cv2.fillPoly(segmentation_mask, [mask], (0, 150+counter*5, 150+counter*5))
                counter+=1

                #only one
                # break

        # Combine the original image with the segmentation mask
        segmentation_result = cv2.addWeighted(image, 0.5, segmentation_mask, 0.7, 0)

        return (segmentation_mask,segmentation_result)

    def get_mask(self,filename):
        # Load the input image using OpenCV
        image = cv2.imread(filename)

        return self.get_mask_from_cvimage(image)


if __name__ == "__main__":
    myyolo = yolo_seg()

    # filename = "44.jpg"
    # mask,result = myyolo.get_mask(filename)
    # # Save the output image with segmentation
    # cv2.imwrite("output_segmentation.jpg", result)
    # # Optionally display the image (make sure you're running in a GUI environment)
    # cv2.imshow("Segmentation Result", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    vf_name = "./videos/3.mp4"
    cap = cv2.VideoCapture(vf_name)  # 输入视频
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            pedestrian_contour = myyolo.get_single_mask_from_cvimage(frame)
            #
            if pedestrian_contour is None:
                continue

            # 转换为灰度图像并进行阈值化处理，膨胀腐蚀操作，提取轮廓
            gray = cv2.cvtColor(pedestrian_contour, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            thresholded = cv2.erode(thresholded, np.ones((3, 3), np.uint8), iterations=1)
            thresholded = cv2.dilate(thresholded, np.ones((2, 2), np.uint8), iterations=1)

            # 将行人轮廓图裁剪成模型需要的大小
            thresholded, flag = myyolo.cut_img(thresholded, 256,256)

            if flag is True:
                continue

            thresholded = thresholded[:, 10: -10].astype('float32') / 255.0

            cv2.imshow("Segmentation Result", thresholded)
            cv2.waitKey(1)
        else:
            break

    cv2.destroyAllWindows()
