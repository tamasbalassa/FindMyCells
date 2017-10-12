import sys
sys.path.append('../AI')
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QFileDialog, QLabel
from PyQt5.QtGui import QFont, QPixmap
from predict_image import predict_one


class MainWindow(QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
               
        global global_image 
        global_image = ''
        
        self.setGeometry(200, 50, 1500, 950)        
        self.initUI()
        
        
    def initUI(self):
        
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setWindowTitle('FindMyCell')
        
        
        # defaul background
        background_image = QLabel(self)
        background_image.setGeometry(150, 50, 400, 200)
        pixmap = QPixmap('..\Utils\FMC.png')
        background_image.setPixmap(pixmap)
        background_image.resize(1200, 820)
        background_image.show()
        
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

        global global_image
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
               
        pic = QLabel(self)        
        pic.setGeometry(150, 50, 400, 200)
        pixmap = QPixmap(fileName)
        pic.setPixmap(pixmap)
        pic.resize(1200, 820)
        pic.show()
        
        global_image = fileName
        
    def predictbutton(self):
                
        global global_image
        if len(global_image) > 0:
            coverage = predict_one(global_image, 1) # TODO parameter
            print coverage
        else:
            print 'NO'
        
        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())