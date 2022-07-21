import cv2
import sys

import numpy as np

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

from tensorflow.keras.models import load_model

class ShowVideo(QtCore.QObject):
    flag = 0

    camera = cv2.VideoCapture(0)

    ret, image = camera.read()
    height, width = image.shape[:2]

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal2 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    def preprocess_input(slef, x):
        x = x.astype('float32')
        x = x / 255.0
        return x

    @QtCore.pyqtSlot()
    def startVideo(self):
        global image,run_video

        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        emotion_classifier = load_model('./models/emotion.hdf5')
        emotion_lables = {0:'angry',1:'blank',2:'funny',3:'sad'}
        while run_video:
            ret, image = self.camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x,y,w,h) in faces:
                face_img = image[y-30:y+h+30, x-30:x+w+30]
                try:
                    face_img = cv2.resize(face_img, (224, 224))
                except:
                    print("except")
                    continue

                face_img = self.preprocess_input(face_img)
                face_img = np.expand_dims(face_img, 0)
                face_img = np.expand_dims(face_img, -1)
                emotion_prediction = emotion_classifier.predict(face_img)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_lables[emotion_label_arg]

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'funny':
                    color = emotion_probability * np.asarray((255, 255, 0))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()

                cv2.rectangle(color_swapped_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(color_swapped_image, emotion_text, (x, y-45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 1, cv2.LINE_AA)

            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                    self.width,
                                    self.height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)


            if self.flag:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_canny = cv2.Canny(img_gray, 50, 100)

                qt_image2 = QtGui.QImage(img_canny.data,
                                         self.width,
                                         self.height,
                                         img_canny.strides[0],
                                         QtGui.QImage.Format_Grayscale8)

                self.VideoSignal2.emit(qt_image2)


            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

class Camera(QWidget):
    def __init__(self):
        global run_video
        super().__init__()
        self.setGeometry(1150, 128, 400, 300)
        self.setWindowTitle("카메라")

        image_viewer=ImageViewer()

        vid.VideoSignal1.connect(image_viewer.setImage)
        btn1 = QPushButton("카메라 확인")

        btn1.clicked.connect(self.startCamera)

        grid = QGridLayout()
        self.setLayout(grid)
        grid.addWidget(image_viewer,0,0)
        grid.addWidget(btn1, 1, 0)

    def startCamera(self):
        vid.startVideo()


class Menu(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(1000, 40, 600, 50)
        self.setWindowTitle("한국인 얼굴 영상 감정 분석 시스템")

        self.window1 = Camera()

        l = QHBoxLayout()

        button1 = QPushButton("카메라 확인")
        button1.clicked.connect(
            lambda checked: self.toggle_window(self.window1)
        )
        l.addWidget(button1)

        button2 = QPushButton("종료")
        button2.clicked.connect(self.Close)
        l.addWidget(button2)

        w = QWidget()
        w.setLayout(l)
        self.setCentralWidget(w)

    def toggle_window(self, window):
        if window.isVisible():
            window.hide()

        else:
            window.show()

    def Close(self):
        widget.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    thread = QtCore.QThread()
    thread.start()
    run_video = True


    vid = ShowVideo()
    vid.moveToThread(thread)

    widget = Menu()

    widget.show()

    app.exec()