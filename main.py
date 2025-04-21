import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QProgressDialog, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,Qt
from mainUI import Ui_MainWindow
from yolo_segment import yolo_seg
from GaitRecognition import GaitRecognition

def get_resource_path(relative_path):
    """ 获取资源文件的正确路径，支持开发环境和打包后环境 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class GaitRecognitionApp(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        # self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('步态识别算法演示平台')
        self.setGeometry(0, 0, 1100, 600)
        self.setFixedSize(1100, 600)

        #for sample and classification
        self.ui.openVideoFolderBtn.clicked.connect(self.extract_features)
        self.ui.openVideoFolderBtn.setEnabled(True)
        self.ui.openSamleDeviceBtn.clicked.connect(self.open_sample_device)
        self.ui.openSamleDeviceBtn.setEnabled(False)
        self.ui.exitBtn.clicked.connect(self.gait_classify)

        # for recognition
        self.ui.openXVideoBtn.clicked.connect(self.open_xvideo)
        self.ui.openXVideoBtn.setEnabled(True)
        self.ui.openCameraBtn.clicked.connect(self.open_cam)
        self.ui.openCameraBtn.setEnabled(False)
        self.ui.gaitRecognizeBtn.clicked.connect(self.gait_recognize)
        self.ui.gaitRecognizeBtn.setEnabled(True)

        #for similarity
        self.ui.openVideo1Btn.clicked.connect(self.open_video_file1)
        self.ui.openVideo1Btn.setEnabled(True)
        self.ui.openVideo2Btn.clicked.connect(self.open_video_file2)
        self.ui.openVideo2Btn.setEnabled(False)
        self.ui.gaitCompareBtn.clicked.connect(self.video_compare)
        self.ui.gaitCompareBtn.setEnabled(True)

        #info
        self.ui.output_msg.setAcceptRichText(False)  # 禁止富文本格式
        self.ui.output_msg.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)  # 启用自动换行

        self.fea_data_path = "./fea_data/"
        self.gallery_folder = "./gallery/"  # 记住的 gallery 文件夹路径
        self.selected_video_path = None
        self.similar_video_path = None

        # 定义计时器，用于循环播放视频
        self.selected_timer = QTimer(self)
        self.selected_timer.timeout.connect(self.play_selected_video)
        self.similar_timer = QTimer(self)
        self.similar_timer.timeout.connect(self.play_similar_video)
        #
        self.video1_timer = QTimer(self)
        self.video1_timer.timeout.connect(self.play_video1)
        self.video2_timer = QTimer(self)
        self.video2_timer.timeout.connect(self.play_video2)

        #for similarity compare
        self.video1_path = None
        self.video2_path = None
        self.original_video1 = None
        self.processed_video1 = None
        self.original_video2 = None
        self.processed_video2 = None

        # detect based on yolo
        self.seger = yolo_seg()

    def gait_classify(self):
        self.showInfo("开始根据视频中的行人将视频分类并保持到不同的分组.....")
        # 显示进度对话框
        progress_dialog = QProgressDialog(self)
        progress_dialog.setWindowTitle("步态视频分组进度")
        progress_dialog.setLabelText("正在进行步态视频分组，请稍候...")
        progress_dialog.setRange(0, 0)  # 设置为无限范围
        progress_dialog.setModal(True)  # 设置为模态对话框，用户不能进行其他操作
        progress_dialog.show()

        # 执行特征提取
        try:
            # 这里通过更新进度回调显示提取进度
            def progress_callback(filename, progress):
                progress_dialog.setLabelText(f"正在处理：{filename}")
                progress_dialog.setValue(progress)
                QApplication.processEvents()  # 更新UI
                self.showInfo("当前处理进度：{}%".format(progress))

            gait_recognition = GaitRecognition()

            # 输入视频文件路径
            target_path = "./videos/分类后"
            feature_file = './videos/fea_list.npy'
            gait_recognition.enroll_or_class_multiple(self.gallery_folder, target_path, feature_file, progress_callback)

        except Exception as e:
            QMessageBox.critical(self, '错误', f'对视频进行分组时出错：\n{e}')
        finally:
            progress_dialog.setValue(100)  # 设置进度为100，表示完成
            progress_dialog.close()  # 关闭进度对话框

    def video_compare(self):
        return #暂时禁用此功能
        # if self.processed_video1 and self.processed_video2:
        #     similarity, same_person = self.gait_discriminator.dis(self.processed_video1, self.processed_video2)
        #     # result_text = f"相似度: {similarity:.2f}\n"
        #     # result_text += "是同一行人" if same_person else "不是同一行人"
        #
        #     result_text = "是同一行人" if same_person else "不是同一行人"
        #     self.showInfo(result_text)
        # else:
        #     self.showStatus("错误! 请先加载两个行人视频！")

    def open_video_file1(self):
        video_path, _ = QFileDialog.getOpenFileName(self, '选择步态视频一', '', 'Video Files (*.mp4 *.avi)')
        if video_path:
            self.showInfo("步态视频一:" + video_path)
            self.ui.video1_title = "步态视频一"
            self.video1_path = video_path
            self.video1_timer.start(33)  # 每 33ms 播放一帧（30 FPS）
            self.original_video1, self.processed_video1 = self.seger.load_video(video_path)

            self.ui.openVideo2Btn.setEnabled(True)
    def open_video_file2(self):
        video_path, _ = QFileDialog.getOpenFileName(self, '选择步态视频二', '', 'Video Files (*.mp4 *.avi)')
        if video_path:
            self.showInfo("步态视频二:" + video_path)
            self.ui.video2_title = "步态视频二"
            self.video2_path = video_path
            self.video2_timer.start(33)  # 每 33ms 播放一帧（30 FPS）
            self.original_video2, self.processed_video2 = self.seger.load_video(video_path)

    def open_cam(self):
        self.showStatus("功能暂时未实现")

    # 采用三个摄像头六个视角，视频编号ID_01 ~ ID_06, 其中ID是8位UID
    # 通过交互式方式让用户输入当前用户信息，并写入pinfo.ini配置文件，实现UID和用户名字的对应
    def open_sample_device(self):
        print("功能暂时未实现")

    def showInfo(self,value):
        cursor = self.ui.output_msg.textCursor()
        cursor.insertText(value + "\n")
        self.ui.output_msg.setTextCursor(cursor)

    def showStatus(self,value):
        self.statusBar().showMessage(value)

    def exitEvent(self):
        #先结束线程，再退出程序
        # if self.lookup_thread!=None and self.lookup_thread.is_alive():
        #     self.stop_event.set()
        #     self.lookup_thread.join()
        app.exit()
    def extract_features(self):
        # 选择 Gallery 文件夹
        folder_path = QFileDialog.getExistingDirectory(self, '选择待处理的步态视频文件夹')
        if folder_path:
            self.showInfo('选择的文件夹为：{}'.format(folder_path))
            self.gallery_folder = folder_path
            # 显示进度对话框
            # progress_dialog = QProgressDialog(self)
            # progress_dialog.setWindowTitle("步态视频分组进度")
            # progress_dialog.setLabelText("正在进行步态视频分组，请稍候...")
            # progress_dialog.setRange(0, 0)  # 设置为无限范围
            # progress_dialog.setModal(True)  # 设置为模态对话框，用户不能进行其他操作
            # progress_dialog.show()
            #
            # # 执行特征提取
            # try:
            #     # 这里通过更新进度回调显示提取进度
            #     def progress_callback(filename, progress):
            #         progress_dialog.setLabelText(f"正在提取：{filename}")
            #         progress_dialog.setValue(progress)
            #         QApplication.processEvents()  # 更新UI
            #
            #     self.gait_recognizer.extract_features_from_folder(folder_path, progress_callback, fea_data_path=self.fea_data_path)
            #     self.gallery_folder = folder_path
            #     self.showInfo('特征提取成功！')
            #     QMessageBox.information(self, '成功',f'特征提取完成，文件保存于：\n {self.fea_data_path}/gallery_features.npy')
            # except Exception as e:
            #     QMessageBox.critical(self, '错误', f'提取特征时出错：\n{e}')
            # finally:
            #     progress_dialog.setValue(100)  # 设置进度为100，表示完成
            #     progress_dialog.close()  # 关闭进度对话框

        else:
            self.showInfo('未选择任何文件夹。')

    def gait_recognize(self):
        # 特征文件位置
        fea_data_filename = os.path.join(os.path.dirname(self.fea_data_path), 'gallery_features.npy')
        # print(fea_data_filename)
        # 判断是否存在默认特征文件
        if not os.path.exists(fea_data_filename):
            QMessageBox.critical(self, '警告', '找不到特征库文件{fea_data_filename}')
            return
        #
        try:
            result = self.gait_recognizer.recognize(self.selected_video_path, fea_data_filename)
            self.showInfo(f'识别完成！')
            self.showInfo(f'命中的Gallery步态视频：{result}')
            self.ui.video2_title.setText("命中的Gallery步态视频")
            self.similar_video_path = os.path.join(self.gallery_folder, result)

            # 开始播放命中的视频
            self.similar_timer.start(33)

        except Exception as e:
            QMessageBox.critical(self, '错误', f'识别时出错：\n{e}')

    def open_xvideo(self):
        video_path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', 'Video Files (*.mp4 *.avi)')
        if video_path:
            self.selected_video_path = video_path
            self.showInfo(f'待识别步态视频：{os.path.basename(video_path)}')
            self.ui.video1_title.setText("待识别步态视频")
            # 开始播放选择的视频
            self.selected_timer.start(33)  # 每 33ms 播放一帧（30 FPS）
        else:
            self.showInfo('未选择任何视频文件。')

    def play_selected_video(self):
        self.play_video(self.selected_video_path, self.ui.video1, self.selected_timer)

    def play_similar_video(self):
        self.play_video(self.similar_video_path, self.ui.video2, self.similar_timer)

    def play_video1(self):
        self.play_video(self.video1_path, self.ui.video1, self.video1_timer)

    def play_video2(self):
        self.play_video(self.video2_path, self.ui.video2, self.video2_timer)
    def play_video(self, video_path, label, timer):
        if not video_path:
            return

        if not hasattr(self, "cap_dict"):
            self.cap_dict = {}

        if video_path not in self.cap_dict:
            self.cap_dict[video_path] = cv2.VideoCapture(video_path)

        cap = self.cap_dict[video_path]
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 如果到视频末尾，则重新播放

    def closeEvent(self, event):
        # 在窗口关闭时释放视频资源
        if hasattr(self, "cap_dict"):
            for cap in self.cap_dict.values():
                cap.release()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GaitRecognitionApp()
    window.show()
    sys.exit(app.exec_())
