import sys
sys.path.append('../AI')
sys.path.append('../Utils')
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QFileDialog, QFormLayout, QHBoxLayout, QGroupBox, QVBoxLayout, QScrollBar
from PyQt5.QtGui import QFont
from predict_image_DIGITS import predict_one
from imageUtils import switch_background, drawRect, set_image
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
        
        vbox2 = QVBoxLayout()
        groupBox2 = QGroupBox("Image", self)
           
        vbox3 = QHBoxLayout(groupBox2)        
        vbox2.addWidget(groupBox2)
        hbox.addLayout(vbox2, 10)        
        
        self.vbox = vbox3        
        
        # defaul background 
        pixmap = set_image(self, '..\Utils\FMC.png', mode='Single')
        switch_background(self, pixmap, mode='Start')
        
        self.s1 = QScrollBar()
        self.s1.setMaximum(255)
        vbox3.addWidget(self.s1, 1)        
        
        self.setLayout(hbox)
        self.show()
        
    def loadbutton(self):
        
        global global_image_path_list
        global_image_path_list = []
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)

        pixmap = set_image(self, fileName, mode='Single')
        switch_background(self, pixmap)
        
        global_image_path_list.append(fileName)
        
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
        self.wheelValue += (delta and delta // abs(delta))
        print(self.wheelValue)

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())