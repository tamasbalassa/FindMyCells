from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPainter, QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QLabel


def switch_background(self, image):
    if (len(self.labels) > 0):
        self.labels[-1].setParent(None)
        self.labels.pop(-1)
    self.labels.append(image)
    self.labels[-1].show()
    
    
def drawRect(pixmap, rectangles):   
        qp = QPainter(pixmap)
        br = QBrush(QColor(100, 10, 10, 40))  
        qp.setBrush(br) 
        for box in rectangles: 
            topleft = QPoint(box[1]-50, box[0]-50)
            bottomright = QPoint(box[1]+50, box[0]+50)
            qp.drawRect(QRect(topleft, bottomright))
        qp.end()


def set_image(self, imgpath, mode='Single'):
        image = QLabel(self)
        image.setGeometry(150, 50, 1200, 820)
        if mode is 'Single':
            pixmap = QPixmap(imgpath)
        elif mode is 'List':
            pixmap = QPixmap(imgpath[-1])
        elif mode is 'Pixmap':
            pixmap = imgpath
            
        pixmap_resized = pixmap.scaled(1200, 820, Qt.KeepAspectRatio)
        image.setPixmap(pixmap_resized)
        return image, pixmap_resized