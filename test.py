import json
import shutil
import time

import uuid
import numpy as np
import torch
from model.network import SetNet
from load import load
import os
import sys
from yolo_segment import yolo_seg

def get_resource_path(relative_path):
    """ 获取资源文件的正确路径，支持开发环境和打包后环境 """
    try:
        # 获取打包后的临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 如果是开发环境中运行，使用当前脚本所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


class GaitRecognition:
    def __init__(self):
        self.sigma = 32
        self.model = SetNet(hidden_dim=256).cuda()

        # 使用 get_resource_path 来获取模型文件路径
        model_path = get_resource_path('./fea_data/rec_model.ptm')
        self.state_dict = torch.load(model_path)
        self.state_dict = {k.replace('module.', ''): v for k, v in self.state_dict.items()}
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

        #detect based on yolo
        self.seger = yolo_seg()

    def load_video_byseger(self,video_path):
        return  self.seger.load_video(video_path)

    def transformer(self, video1, video2=None):
        video1 = torch.tensor(np.array(video1)).unsqueeze(0)
        if video2 is not None:
            video2 = torch.tensor(np.array(video2)).unsqueeze(0)
            return video1, video2
        return video1

    def get_feature(self, video):
        frames = torch.tensor([[video.size(1)]])
        feature = self.model(video.cuda(), frames.cuda())[0].flatten()
        return feature.data.cpu().numpy()

    def load_gallery(self, gallery_path):
        # 使用 get_resource_path 获取特征库文件路径
        gallery_path = get_resource_path(gallery_path)
        feature_list = np.load(gallery_path, allow_pickle=True)
        return feature_list

    def recognize(self, video_path, gallery_path,thresh=0.5):
        # 使用 get_resource_path 获取视频文件路径
        video_path = get_resource_path(video_path)
        # 提取新视频的特征
        # _, video = load(video_path)
        _, video = self.seger.load_video(video_path)

        video_tensor = self.transformer(video)
        feature = self.get_feature(video_tensor)

        # 加载特征库
        gallery_features = self.load_gallery(gallery_path)

        max_similarity = -1
        best_match = None

        # 遍历特征库，计算与每个特征的相似度
        for feature_in_gallery, clsid in gallery_features:
            similarity = self.get_similarity(feature, feature_in_gallery.flatten())

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = clsid

        print("\n max_similarity:",max_similarity)
        if max_similarity>thresh:
            return best_match
        else:
            return None
    def get_similarity(self, feature1, feature2):
        # 计算欧氏距离相似度
        distance = np.linalg.norm(feature1 - feature2)
        return np.exp(-(distance ** 2) / (2 * self.sigma ** 2))

    def dis(self, video1, video2, thresh=0.5):
        video1,video2 = self.transformer(video1,video2)
        feature1 = self.get_feature(video1)
        feature2 = self.get_feature(video2)
        similarity = self.get_similarity(feature1, feature2)
        print("similarity:", similarity)
        if similarity > thresh:
            return similarity, True
        else:
            return similarity, False

    def extract_features_from_folder(self, folder_path, progress_callback=None, fea_data_path=None):
        save_path = None
        feature_list = []
        total_files = len([f for f in os.listdir(folder_path) if f.endswith('.mp4') or f.endswith('.avi')])

        # 遍历文件夹中的所有视频文件
        for idx, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith('.mp4') or filename.endswith('.avi'):
                video_path = os.path.join(folder_path, filename)
                # _, video = load(video_path)
                _, video = self.seger.load_video(video_path)

                # 提取视频特征
                video_tensor = self.transformer(video)
                feature = self.get_feature(video_tensor)

                # 将特征和文件名组合为 [特征, 文件名]
                feature_with_name = [feature, filename]
                feature_list.append(feature_with_name)

                # 更新进度
                if progress_callback:
                    progress = int(((idx + 1) / total_files) * 100)
                    progress_callback(filename, progress)

        # 将所有特征合并成一个二维数组
        if feature_list:
            all_features = np.array(feature_list, dtype=object)  # 使用object类型来处理不同类型的数据

            #如果指定了fea_data文件路径则：
            if fea_data_path:
                if not os.path.exists(fea_data_path):
                    os.mkdir(fea_data_path)
                save_path = os.path.join(fea_data_path, 'gallery_features.npy')  # 保存到父文件夹
                print(save_path)
            else:
                # 获取上一级文件夹的路径，并构造保存路径
                parent_folder = os.path.dirname(os.path.abspath(folder_path))  # 获取上一级文件夹的绝对路径
                save_path = os.path.join(parent_folder, 'gallery_features.npy')  # 保存到父文件夹

            np.save(save_path, all_features)
            print(f"Features saved to '{save_path}'")
        else:
            print("No valid video files found.")

        return save_path

    def get_10bit_uid(self):
        # 生成基于时间戳和随机数的唯一 ID
        unique_id = uuid.uuid1().int >> 64
        return abs(unique_id % 10000000000)

    def enroll_or_class_multiple(self,vpath,tpath,ffile,progress_callback=None):
        #
        total_files = len([f for f in os.listdir(vpath) if f.endswith('.mp4') or f.endswith('.avi')])
        #
        for idx, filename in enumerate(os.listdir(vpath)):
        # for filename in os.listdir(vpath):
            filepath = os.path.join(vpath, filename)
            if os.path.isfile(filepath):  # 确保是文件而不是子目录
                # print("\n 当前在处理：{}".format(filepath))
                self.enroll_or_class_one(filepath,tpath,ffile)

            # 更新进度
            if progress_callback:
                progress = int(((idx + 1) / total_files) * 100)
                progress_callback(filename, progress)
    def enroll_or_class_one(self,vfile,tpath,ffile):
        fea_list=[]
        #处理逻辑，分为两种情况
        #1. vfile不存在，则创建一个ffile; 提取vfile特征vx；创建一个具有唯一ClassID的文件夹并将vfile拷贝进去；同时将[classID,vx]写入ffile
        #2. vfile已存在，则直接调用recognize函数，返回匹配的classID或者None（根据调用时设置的threshhold参数确定，默认为0.5）；然后进一步分为两种情况：
        #2.1 返回classID，则将vfile拷入classID对应的文件夹
        #2.2 返回None，则新生成一个具有唯一ClassID的文件夹并将vfile拷贝进去；同时，提取vfile特征vx，并将[classID,vx]写入ffile

        ######################################
        #提取vfile特征vx
        # print("\n Start load \n")
        _, video = self.seger.load_video(vfile)
        if len(video)<1:
            print("未检测到有效步态，忽略该视频。")
            return
        # print("\n Load finished.")

        video_tensor = self.transformer(video)
        new_fea = self.get_feature(video_tensor)
        new_classid = self.get_10bit_uid()
        feature_with_cls = [new_fea, new_classid]
        ######################################
        if not os.path.exists(ffile):  # 如果文件不存在
            fea_list.append(feature_with_cls)
            fea_arr = np.array(fea_list, dtype=object)
            np.save(ffile,fea_arr)
            #
            # 创建以 classID 命名的子文件夹
            target_folder = os.path.join(tpath, str(new_classid))
            os.makedirs(target_folder, exist_ok=True)  # 确保文件夹存在

            # 复制文件到子文件夹，保留原文件名和扩展名
            target_file = os.path.join(target_folder, os.path.basename(vfile))
            shutil.copy(vfile, target_file)
        else:
            best_match = self.recognize(vfile, ffile, thresh=0.85)
            if best_match is not None: # 2.1 #not None
                print("当前视频匹配的步态类为{}。".format(best_match))
                target_path = os.path.join(tpath,str(best_match))
                os.makedirs(target_path, exist_ok=True)  # 确保文件夹存在

                # 复制文件到子文件夹，保留原文件名和扩展名
                target_file = os.path.join(target_path, os.path.basename(vfile))
                shutil.copy(vfile, target_file)
            else: # 2.2 #None
                print("当前视频找不到匹配步态，新建步态类。")
                exist_feat_arr = self.load_gallery(ffile)
                exist_feat_list = exist_feat_arr.tolist()
                exist_feat_list.append(feature_with_cls)
                temp_arr = np.array(exist_feat_list, dtype=object)
                np.save(ffile, temp_arr)
                #
                target_folder = os.path.join(tpath, str(new_classid))
                os.makedirs(target_folder, exist_ok=True)  # 确保文件夹存在

                # 复制文件到子文件夹，保留原文件名和扩展名
                target_file = os.path.join(target_folder, os.path.basename(vfile))
                shutil.copy(vfile, target_file)

        # 进行识别，输出最相似的视频文件名

    def compare_two_videos(video_path1, video_path2, thresh=0.5):
        gait_recognition = GaitRecognition()

        # 加载并处理两个视频
        _, video1 = gait_recognition.seger.load_video(video_path1)
        _, video2 = gait_recognition.seger.load_video(video_path2)

        # 计算相似度并返回结果
        similarity, is_match = gait_recognition.dis(video1, video2, thresh)

        print(f"视频 {video_path1} 和 {video_path2} 的相似度为: {similarity}")
        print(f"是否匹配: {is_match}")

        return similarity, is_match


'''if __name__ == '__main__':
    # 关闭所有级别的日志输出
    # import logging
    # logging.disable(logging.CRITICAL)

    gait_recognition = GaitRecognition()

    # 输入视频文件路径
    target_path = "./videos/分类后"
    feature_file = './videos/fea_list.npy'
    video_path= "./videos/分类前"
    gait_recognition.enroll_or_class_multiple(video_path,target_path,feature_file)
'''
if __name__ == '__main__':
    video_path1 = r'C:\Users\dy\Desktop\gaitClassify\videos\3211608560\001-04-1.mp4'
    video_path2 = r'C:\Users\dy\Desktop\gaitClassify\videos\3211608560\001-06-1.mp4'

    # 使用方法一
    similarity1, is_match1 = GaitRecognition.compare_two_videos(video_path1, video_path2, thresh=0.85)

    # 或使用方法二
    # similarity2, is_match2 = compare_videos_step_by_step(video_path1, video_path2, thresh=0.7)




