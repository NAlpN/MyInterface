import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QListWidget
from PyQt5.QtCore import QTimer, Qt
from Arayuz import Ui_MainWindow
from ultralytics import YOLO
import cv2
from goruntu import Goruntu

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.mesajlar_widget = self.ui.Mesajlar
        
        pixmap = QPixmap("RA.png")
        self.ui.Logo.setPixmap(pixmap)

        pixmap_resized = pixmap.scaled(self.ui.Logo.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.Logo.setPixmap(pixmap_resized)
        self.ui.Logo.setAlignment(Qt.AlignCenter)

        self.timer = QTimer()
        self.timer.timeout.connect(self.goruntu_guncelle)
        self.camera = None

        self.ui.KameraAc.clicked.connect(self.kamera_ac)
        self.ui.KameraKapat.clicked.connect(self.kamera_kapat)
        self.ui.NesneTanima.clicked.connect(self.timer2)
        self.ui.FotografCek.clicked.connect(self.fotograf_cek)
        self.ui.FotografEkle.clicked.connect(self.gorsel_sec)
        self.ui.VideoEkle.clicked.connect(self.video_sec)

        self.arkaPlan(QColor('#E6E6FA'))

    def arkaPlan(self, color):
        renk = self.palette()
        renk.setColor(self.backgroundRole(), color)
        self.setPalette(renk)

    def kamera_ac(self):
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            QtWidgets.QMessageBox.critical(self, "Hata", "Kamera açılamadı. Lütfen kameranın bağlı olduğundan emin olun.")
            return

        self.ui.KameraGoruntu.setAlignment(Qt.AlignCenter)
        self.timer.start(30)
        self.mesajlar_widget.addItem("Kamera açıldı.")
    
    def kamera_kapat(self):
        if self.camera is not None:
            self.camera.release()
            self.timer.stop()
            self.ui.KameraGoruntu.clear()
            self.mesajlar_widget.addItem("Kamera kapatıldı.")

    def timer2(self):
        self.timer3 = QTimer(self)
        self.timer3.timeout.connect(self.nesne_tanima)
        self.timer3.start(3)

    def nesne_tanima(self):
        self.nesnetanima = Goruntu()
        qImg = self.nesnetanima.run()
        self.ui.KameraGoruntu.setAlignment(Qt.AlignCenter)
        self.ui.KameraGoruntu.setPixmap(QPixmap.fromImage(qImg))

    def goruntu_guncelle(self):
        ret, frame = self.camera.read()
        if ret:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            self.ui.KameraGoruntu.setPixmap(QPixmap.fromImage(p))
    
    def fotograf_cek(self):
        if self.camera is not None:
            dosya_adi, _ = QFileDialog.getSaveFileName(self, "Fotoğrafı Kaydet", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")

            if dosya_adi:
                if not dosya_adi.endswith(('.png', '.jpg', '.jpeg')):
                    dosya_adi += '.png'

                ret, frame = self.camera.read()
                if ret:
                    cv2.imwrite(dosya_adi, frame)
                    self.mesajlar_widget.addItem("Fotoğraf kaydedildi.")
                else:
                    self.mesajlar_widget.addItem("Fotoğraf keydidilemedi.")
            else:
                self.mesajlar_widget.addItem("Dosya adı belirtilmedi veya geçersiz bir dosya adı belirtildi.")
    
    def gorsel_sec(self):
        dosya_adi, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Images (*.png *.jpg *.jpeg)")
        model = YOLO("yolov8s.pt")
        if dosya_adi:
            sonuc = model.predict(source = dosya_adi, save = True)
            dosya_adlari = [os.path.basename(item.path) for item in sonuc]
            print(dosya_adlari)
            img = QImage(f"run/detect/predict/{dosya_adlari[0]}")
            self.ui.KameraGoruntu.setAlignment(Qt.AlignCenter)
            self.ui.KameraGoruntu.setPixmap(QPixmap.fromImage(img))
            self.mesajlar_widget.addItem("Görsel seçildi")
        else:
            self.mesajlar_widget.addItem("Dosya okunamadı.")

    def video_sec(self):
        dosya_adi, _ = QFileDialog.getOpenFileName(self, "Video Seç", "", "Videos (*.mp4)")

        if dosya_adi:
            self.mesajlar_widget.addItem("Video seçildi")
        else:
            self.mesajlar_widget.addItem("Dosya okunamadı.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.setFixedSize(1024, 768)
    mainWindow.show()
    sys.exit(app.exec_())