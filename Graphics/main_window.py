import sys
sys.path.append('../AI')
sys.path.append('../Utils')
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QFileDialog, QFormLayout, QHBoxLayout, QGroupBox, QVBoxLayout, QScrollBar
from PyQt5.QtGui import QFont
from predict_image_DIGITS import predict_one
from imageUtils import switch_background, drawRect, set_image, load_images_from_dir, scroll_image
import settings


class MainWindow(QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
               
        global global_image_path_list
        global_image_path_list = []
        
        self.wheelValue = 0
        
        settings.init()
        
        self.setGeometry(200, 50, 1500, 950)
        self.labels = []
        self.initUI()
        
        
    def initUI(self):
        
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setWindowTitle('FindMyCell')
        
        hbox = QHBoxLayout()
        groupBox1 = QGroupBox("Menu", self)
        vbox = QVBoxLayout(groupBox1)
        flay = QFormLayout()
        load_btn = QPushButton('Load', self)
        load_btn.clicked.connect(self.loadbutton)
        flay.addRow(load_btn)
        vbox.addWidget(load_btn, 1)
        
        predict_btn = QPushButton('Predict', self)
        predict_btn.clicked.connect(self.predictbutton)
        flay.addRow(predict_btn)
        vbox.addWidget(predict_btn, 1)        
        vbox.addLayout(flay, 1)
        
        hbox.addWidget(groupBox1, 1)
        
        vbox2 = QHBoxLayout()
        groupBox2 = QGroupBox("Image", self)
           
        vbox3 = QHBoxLayout(groupBox2)        
        vbox2.addWidget(groupBox2)
        hbox.addLayout(vbox2, 10)        
        
        self.vbox = vbox3        
        
        # defaul background 
        pixmap = set_image(self, '..\Utils\FMC.png', mode='Single')
        switch_background(self, pixmap, mode='Start')
        
        self.s1 = QScrollBar()
        self.setSliderMax(1)
# FUTURE RELEASE
#        self.s1.sliderMoved.connect(self.s1.setValue)
        vbox2.addWidget(self.s1, 1)        
        
        self.setLayout(hbox)
        self.show()
        
    def loadbutton(self):
        
        global global_image_path_list
        global_image_path_list = []
        
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        #fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        imgdir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        global_image_path_list = load_images_from_dir(self, imgdir, global_image_path_list)
        
        self.setSliderMax(len(global_image_path_list))

        print self.s1.maximum()

        #pixmap = set_image(self, fileName, mode='Single')
        #switch_background(self, pixmap)
        
        #global_image_path_list.append(fileName)
        
    def predictbutton(self):       
        
        pixmapToPredict = self.vbox.itemAt(0).widget().pixmap()
        if (type(global_image_path_list) is list):
            rectangles, regprops = predict_one(global_image_path_list, 1) # TODO parameter
            drawRect(pixmapToPredict, rectangles)
        
            pm = set_image(self, pixmapToPredict, mode='Pixmap')
            switch_background(self, pm)
            
        print regprops
            
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if ((self.wheelValue - (delta and delta // abs(delta))) >= 0) and ((self.wheelValue - (delta and delta // abs(delta))) <= self.s1.maximum()):
            self.wheelValue -= (delta and delta // abs(delta))
        print(self.wheelValue)
        self.s1.setValue(self.wheelValue)
        
        global global_image_path_list
        if (type(global_image_path_list) is list) and len(global_image_path_list) > 0:
            global_image_path_list = scroll_image(self, global_image_path_list)
# FUTURE RELEASE
# =============================================================================
#         self.setSliderValue(self.wheelValue)
# 
#     def setSliderValue(self, value):
#         self.s1.setValue(value)
# =============================================================================

    def setSliderMax(self, maxValue):
        self.s1.setMaximum(maxValue-1)

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())