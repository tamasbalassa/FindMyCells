import sys
sys.path.append('../Utils')
import settings
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPainter, QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox


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


def set_image(self, imgpath, mode='Single'):                
        
        if mode is 'Single':
            pixmap = QPixmap(imgpath)
        elif mode is 'List':
            pixmap = QPixmap(imgpath[0])
        elif mode is 'Pixmap':
            pixmap = imgpath
            
        pixmap_resized = pixmap.scaled(settings.gWidth, settings.gHeight, Qt.KeepAspectRatio)
        
        return pixmap_resized

def load_images_from_dir(self, imagesPath):
       
    c = 0
    for filepath in imagesPath:
        self.global_image_path_list.append(filepath)
        c += 1
    
    if c < 1:
           QMessageBox.about(self, "ERROR", "Please select at least one image.")
    else:
        pixmap = set_image(self, self.global_image_path_list[0], mode='Single')
        switch_background(self, pixmap)
            
def scroll_image(self, global_image_path_list):
        
    fileName = global_image_path_list[self.wheelValue]
    pixmap = set_image(self, fileName, mode='Single')
    switch_background(self, pixmap)
    
    return global_image_path_list