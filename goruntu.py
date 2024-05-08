from PyQt5.QtCore import QTimer, QObject
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import cv2
import sys
from ultralytics import YOLO

class Goruntu(QObject):
    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8s.pt')
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run)
        self.timer.start(10)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                results = self.model(frame)
                if results is not None and results[0].boxes is not None:
                    annotated_frame = frame.copy()
                    annotated_frame = results[0].plot()

                    yukseklik, genislik, kanal = annotated_frame.shape
                    step = kanal * genislik

                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    qImg = QImage(annotated_frame.data, genislik, yukseklik, step, QImage.Format_RGB888)

                    return qImg

        self.cap.release()
        cv2.destroyAllWindows()