import sys
sys.path.append('../AI')
sys.path.append('../Utils')
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QFileDialog
from PyQt5.QtGui import QFont
from predict_image_DIGITS import predict_one
from imageUtils import switch_background, drawRect, set_image


class MainWindow(QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
               
        global global_image_path_list, global_pixmap_list 
        global_image_path_list = []
        global_pixmap_list = []
        
        self.setGeometry(200, 50, 1500, 950)
        self.labels = []
        self.initUI()
        
        
    def initUI(self):
        
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setWindowTitle('FindMyCell')
        
        
        # defaul background        
        background_image, pixmap = set_image(self, '..\Utils\FMC.png', mode='Single')
        switch_background(self, background_image)
        
        # load button
        load_btn = QPushButton('Load', self)
        load_btn.clicked.connect(self.loadbutton)
        load_btn.resize(load_btn.sizeHint())
        load_btn.move(28, 150)    
        
        # predict button
        predict_btn = QPushButton('Predict', self)
        predict_btn.clicked.connect(self.predictbutton)
        predict_btn.resize(predict_btn.sizeHint())
        predict_btn.move(28, 200)
        
        
        self.show()
        
    def loadbutton(self):      

        global global_image_path_list, global_pixmap_list
        global_image_path_list = []
        global_pixmap_list = []
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        
        
        pic, pixmap = set_image(self, fileName, mode='Single')
        switch_background(self, pic)
        
        global_image_path_list.append(fileName)
        global_pixmap_list.append(pixmap)
        
    def predictbutton(self):
                
        global global_image_path_list, global_pixmap_list
        if (type(global_image_path_list) is list) and (type(global_pixmap_list) is list):
            
            rectangles, regprops = predict_one(global_image_path_list, 1) # TODO parameter
            drawRect(global_pixmap_list[-1], regprops)
            
            pic, pm = set_image(self, global_pixmap_list[-1], mode='Pixmap')
            switch_background(self, pic)
            
            print regprops
        else:
            print 'NO prediction'
        
        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())