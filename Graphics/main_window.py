import sys
sys.path.append('../AI')
sys.path.append('../Utils')
import numpy as np
import csv
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QFileDialog, QFormLayout, QHBoxLayout, QGroupBox, QVBoxLayout, QScrollBar
from PyQt5.QtGui import QFont
from predict_image_DIGITS import predict_one, predict_all
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

# =============================================================================        
        # menu group starts here
        groupBox1 = QGroupBox("Menu", self)
        vbox = QVBoxLayout(groupBox1)
        flay = QFormLayout()
        
        # load button
        load_btn = QPushButton('Load', self)
        load_btn.clicked.connect(self.loadbutton)
        flay.addRow(load_btn)
        vbox.addWidget(load_btn, 1)
        
        # predict button
        predict_btn = QPushButton('Predict', self)
        predict_btn.clicked.connect(self.predictbutton)
        flay.addRow(predict_btn)
        vbox.addWidget(predict_btn, 1)        
        
        # statistics button
        stat_btn = QPushButton('Create stats', self)
        stat_btn.clicked.connect(self.statisticsbutton)
        flay.addRow(stat_btn)
        vbox.addWidget(stat_btn, 1)
        
        # switch model button
        switch_model_btn = QPushButton('Switch model', self)
        switch_model_btn.clicked.connect(self.switchmodelbutton)
        flay.addRow(switch_model_btn)
        vbox.addWidget(switch_model_btn, 1)
        
        vbox.addLayout(flay, 1)        
        hbox.addWidget(groupBox1, 1)
        
        # menu group ends here        
# =============================================================================
        
        # image group starts here
        vbox2 = QHBoxLayout()
        groupBox2 = QGroupBox("Image", self)
           
        vbox3 = QHBoxLayout(groupBox2)        
        vbox2.addWidget(groupBox2)
        hbox.addLayout(vbox2, 10)        
        
        self.vbox = vbox3 # need this to be able to switch the image        
        
        # defaul background 
        pixmap = set_image(self, '..\Utils\FMC.png', mode='Single')
        switch_background(self, pixmap, mode='Start')
        
        # scrollbar
        self.scrollbar = QScrollBar()
        self.setSliderMax(1)
# FUTURE RELEASE
#        self.scrollbar.sliderMoved.connect(self.scrollbar.setValue)
        vbox2.addWidget(self.scrollbar, 1)        

        # image group ends here
# =============================================================================   
     
        self.setLayout(hbox)
        self.show()
        
    def loadbutton(self):
        
        global global_image_path_list
        global_image_path_list = []

        imgdir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        global_image_path_list = load_images_from_dir(self, imgdir, global_image_path_list)
        
        self.setSliderMax(len(global_image_path_list))
        
    def predictbutton(self):       
        
        pixmapToPredict = self.vbox.itemAt(0).widget().pixmap()
        if (type(global_image_path_list) is list):
            rectangles = predict_one([global_image_path_list[self.wheelValue]], 1) # TODO parameter
            drawRect(pixmapToPredict, rectangles)
        
            pm = set_image(self, pixmapToPredict, mode='Pixmap')
            switch_background(self, pm)
       
    def statisticsbutton(self):
        global global_image_path_list
        
        rectangles = predict_all(global_image_path_list, 1)
        
        with open('eggs.csv', 'wb') as csvfile:
            fieldnames = ['Image Number', 'Cell ID', 'x1', 'y1', 'x2', 'y2']
            writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)        
            writer.writeheader()
            
            cellID = 0
            imgID = 0
            for row in rectangles:
                if row[-1] != imgID:
                    imgID = row[-1]
                    cellID = 0
                else:
                    cellID += 1
                writer.writerow({'Image Number': row[-1], 'Cell ID': cellID, 'x1': int(row[0]), 'y1': int(row[1]), 'x2': int(row[2]), 'y2': int(row[3])})
                   
# =============================================================================
#         with file('test2.txt', 'w') as outfile:
#             outfile.write('# Total number of cells: {0}\n'.format(len(rectangles)))
#             outfile.write('\n# Image number: 0\n')
#             outfile.write('# =======================================================\n')
#             outfile.write('# Bounding box positions:\n')
#             imgID = 0
#             for row in rectangles:
#                 if row[-1] != imgID:
#                     imgID = row[-1]
#                     outfile.write('\n# Image number: {0}\n'.format(imgID))
#                     outfile.write('# =======================================================\n')                    
#                     outfile.write('# Bounding box positions: \n')
#                     
#                 np.savetxt(outfile, [row[:-1]], fmt='%-7.0f')
# =============================================================================
        
    def switchmodelbutton(self):
        print 'model'
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if ((self.wheelValue - (delta and delta // abs(delta))) >= 0) and ((self.wheelValue - (delta and delta // abs(delta))) <= self.scrollbar.maximum()):
            self.wheelValue -= (delta and delta // abs(delta))
        self.scrollbar.setValue(self.wheelValue)
        print(self.wheelValue)
        
        global global_image_path_list
        if (type(global_image_path_list) is list) and len(global_image_path_list) > 0:
            global_image_path_list = scroll_image(self, global_image_path_list)
# FUTURE RELEASE
# =============================================================================
#         self.setSliderValue(self.wheelValue)
# 
#     def setSliderValue(self, value):
#         self.scrollbar.setValue(value)
# =============================================================================

    def setSliderMax(self, maxValue):
        self.scrollbar.setMaximum(maxValue-1)

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())