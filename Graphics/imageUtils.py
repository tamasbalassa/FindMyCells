import sys
sys.path.append('../Utils')
import settings
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPainter, QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QLabel


def switch_background(self, pixmap, mode='None'):
    if (mode is not 'Start') and (len(self.vbox) > 0):
        self.vbox.itemAt(0).widget().setParent(None)
    label = QLabel(self)
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)
    self.vbox.addWidget(label, 0)
    
def drawRect(pixmap, rectangles):   
        qp = QPainter(pixmap)
        xScale = pixmap.width() / float(settings.gInputWidth)
        yScale = pixmap.height() / float(settings.gInputHeight)
        br = QBrush(QColor(100, 10, 10, 40))  
        qp.setBrush(br) 
        for box in rectangles: 
            
            if box[0] < 1: box[0] = 0
            if box[1] < 1: box[1] = 0
            if box[2] > 703: box[2] = 703
            if box[3] > 511: box[3] = 511
            
            topleft = QPoint(box[0] * xScale, box[1] * yScale)
            bottomright = QPoint(box[2] * xScale, box[3] * yScale)
            qp.drawRect(QRect(topleft, bottomright))
        
        qp.end()


def set_image(self, imgpath, mode='Single', image=0):                
        
        if mode is 'Single':
            pixmap = QPixmap(imgpath)
        elif mode is 'List':
            pixmap = QPixmap(imgpath[-1])
        elif mode is 'Pixmap':
            pixmap = imgpath
            
        pixmap_resized = pixmap.scaled(settings.gWidth, settings.gHeight, Qt.KeepAspectRatio)
        
        return pixmap_resized
