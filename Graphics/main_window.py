import sys, os
sys.path.append('../AI')
sys.path.append('../Utils')
import csv
#from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QFileDialog, QFormLayout, QHBoxLayout, QGroupBox, QVBoxLayout, QScrollBar
import PyQt5.QtCore as qc
from PyQt5.QtCore import Qt
import PyQt5.QtWidgets as wdg
from PyQt5.QtGui import QFont
from predict_image_DIGITS import predict_one, predict_all
from imageUtils import switch_background, drawRect, set_image, load_images_from_dir, scroll_image
import settings


class MainWindow(wdg.QWidget):
    
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
        
        wdg.QToolTip.setFont(QFont('SansSerif', 10))
        self.setWindowTitle('FindMyCell')
        
        hbox = wdg.QHBoxLayout()

# =============================================================================        
        # menu group starts here
        groupBox1 = wdg.QGroupBox("Menu", self)
        vbox = wdg.QVBoxLayout(groupBox1)
        flay1 = wdg.QFormLayout()
        
        vbox.addSpacing(30)
        
        # LOAD AND PREDICT SECTION
        groupBox1_a = wdg.QGroupBox("L&P", self)
        vbox.addWidget(groupBox1_a)
        vbox1_a = wdg.QVBoxLayout(groupBox1_a)
        
        # load button
        load_btn = wdg.QPushButton('Load', self)
        load_btn.clicked.connect(self.loadbutton)
        flay1.addRow(load_btn)
        vbox1_a.addWidget(load_btn, 1, Qt.AlignTop)
        
        # predict button
        predict_btn = wdg.QPushButton('Predict', self)
        predict_btn.clicked.connect(self.predictbutton)
        flay1.addRow(predict_btn)
        vbox1_a.addWidget(predict_btn, 1, Qt.AlignTop)

        vbox1_a.addLayout(flay1, 1)
         
        vbox.addSpacing(100)
        
        # UTILS SECTION
        flay_utils = wdg.QFormLayout()
        groupBox1_utils = wdg.QGroupBox("UTILS", self)
        vbox.addWidget(groupBox1_utils)
        vbox1_utils = wdg.QVBoxLayout(groupBox1_utils)
        
        # switch model button
        #switch_model_btn = wdg.QPushButton('Switch model', self)
        #switch_model_btn.clicked.connect(self.switchmodelbutton)
        #flay_utils.addRow(switch_model_btn)
        #vbox1_utils.addWidget(switch_model_btn, 1)
        
        # switch model text
        switch_model_text = wdg.QLabel()
        switch_model_text.setText("Select Model:")
        flay_utils.addRow(switch_model_text)
        vbox1_utils.addWidget(switch_model_text, 1)
        
        # switch model combobutton
        combo_model = wdg.QComboBox(self)
        flay_utils.addRow(combo_model)
        vbox1_utils.addWidget(combo_model, 1)
        
        vbox1_utils.addSpacing(20)
        
        # select image type text
        select_img_type = wdg.QLabel()
        select_img_type.setText("Image Type:")
        flay_utils.addRow(select_img_type)
        vbox1_utils.addWidget(select_img_type, 1)       
        
        
        # select image type combobutton
        combo_imgtype = wdg.QComboBox(self)
        flay_utils.addRow(combo_imgtype)
        vbox1_utils.addWidget(combo_imgtype, 1)
        
        vbox1_utils.addLayout(flay_utils, 1)
        vbox1_utils.insertSpacing(0, 20)
        vbox1_utils.insertSpacing(5, 2)
        
        vbox.addSpacing(100)
        
        # OUTPUT GROUP        
        flay_output = wdg.QFormLayout()
        
        groupBox1_output = wdg.QGroupBox("OUTPUT", self)
        vbox.addWidget(groupBox1_output)
        vbox1_output = wdg.QVBoxLayout(groupBox1_output)
        
        # statistics button
        stat_btn = wdg.QPushButton('Create stats', self)
        stat_btn.clicked.connect(self.statisticsbutton)
        flay_output.addRow(stat_btn)
        vbox1_output.addWidget(stat_btn, 1)
        
        vbox.addSpacing(150)
        
        #vbox.addLayout(flay, 1)        
        hbox.addWidget(groupBox1, 1)
        
        # menu group ends here        
# =============================================================================
        
        # image group starts here
        vbox2 = wdg.QHBoxLayout()
        groupBox2 = wdg.QGroupBox("Image", self)
           
        vbox3 = wdg.QHBoxLayout(groupBox2)        
        vbox2.addWidget(groupBox2)
        hbox.addLayout(vbox2, 10)        
        
        self.vbox = vbox3 # need this to be able to switch the image        
        
        # defaul background 
        pixmap = set_image(self, '..\Utils\FMC.png', mode='Single')
        switch_background(self, pixmap, mode='Start')
        
        # scrollbar
        self.scrollbar = wdg.QScrollBar()
        self.setSliderMax(1)
# FUTURE RELEASE
#        self.scrollbar.sliderMoved.connect(self.scrollbar.setValue)
        vbox2.addWidget(self.scrollbar, 1)        

        # image group ends here
# =============================================================================   
     
        self.setLayout(hbox)
        self.show()
        
    def loadbutton(self):
        
        global global_image_path_list, imgdir
        global_image_path_list = []

        imgdir = str(wdg.QFileDialog.getExistingDirectory(self, "Select Directory"))
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
        global global_image_path_list, imgdir
        
        rectangles = predict_all(global_image_path_list, 1)
        
        with open('eggs.csv', 'wb') as csvfile:
            fieldnames = ['Image Number', 'Cell ID', 'x1', 'y1', 'x2', 'y2']
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)        
            writer.writeheader()
            
            cellID = 0
            imgID = 0
            for row in rectangles:
                if row[-1] != imgID:
                    imgID = row[-1]
                    cellID = 0
                else:
                    cellID += 1
                imgName = global_image_path_list[row[-1]].split(os.sep)
                
                writer.writerow({'Image Number': imgName[-1], 'Cell ID': cellID, 'x1': int(row[0]), 'y1': int(row[1]), 'x2': int(row[2]), 'y2': int(row[3])})
                   
        
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
    
    app = wdg.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())