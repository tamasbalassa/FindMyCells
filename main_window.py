import sys, os, csv
import PyQt5.QtWidgets as wdg
import PyQt5.QtGui as qg
from PyQt5.QtCore import Qt
import argparse

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
#from predict_image_DIGITS import predict_one, predict_all
#from ai_utils import change_params_in_deploy
#from graphics.image_utils import switch_background, drawRect, set_image, load_images_from_dir, scroll_image
from graphics.image_utils import *
from util.FMC_settings import *

from AI.torch_utils import predict_image_torch

parser_predict = argparse.ArgumentParser()
#parser_predict.add_argument('--image_folder', type=str, default='samples', help='path to dataset')
parser_predict.add_argument('--config_path', type=str, default='AI/configs/yolov3.cfg', help='path to model config file')
parser_predict.add_argument('--weights_path', type=str, default='AI/models/yolov3.weights', help='path to weights file')
parser_predict.add_argument('--class_path', type=str, default='coco.names', help='path to class label file')
parser_predict.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser_predict.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser_predict.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser_predict.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser_predict.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser_predict.add_argument('--use_cuda', type=bool, default=False, help='whether to use cuda if available')
opt_predict = parser_predict.parse_args()

class MainWindow(wdg.QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
               
        self.global_image_path_list = []
        
        self.wheelValue = 0

        FMC_init()
        
        self.setGeometry(200, 50, 1500, 950)
        self.labels = []
        self.initUI()
        
        
    def initUI(self):
        
        wdg.QToolTip.setFont(qg.QFont('SansSerif', 10))
        self.setWindowTitle('FindMyCells')
        
        hbox = wdg.QHBoxLayout()

# =============================================================================        
        # menu group starts here
        groupBox1 = wdg.QGroupBox("Menu", self)
        vbox = wdg.QVBoxLayout(groupBox1)
        
        vbox.addSpacing(50)
        
        # INPUT SECTION
        groupBox1_input = wdg.QGroupBox("INPUT",self)
        vbox.addWidget(groupBox1_input)
        vbox1_input = wdg.QVBoxLayout(groupBox1_input)
        vbox1_input.setContentsMargins(10, 5, 10, 5)        
        vbox1_input.setSpacing(0)
        
        # load button        
        vbox1_input.addSpacing(10)
        load_btn = wdg.QPushButton('Load', self)
        load_btn.clicked.connect(self.loadbutton)
        vbox1_input.addWidget(load_btn, 1, Qt.AlignTop)
        vbox1_input.addSpacing(10)
                        
        vbox.addSpacing(80)
        
        # SETTINGS SECTION
        groupBox1_utils = wdg.QGroupBox("SETTINGS", self)
        vbox.addWidget(groupBox1_utils)
        vbox1_utils = wdg.QVBoxLayout(groupBox1_utils)
        vbox1_utils.setContentsMargins(10, 5, 10, 5)
        vbox1_utils.setSpacing(0)
        vbox1_utils.addSpacing(10)
        
        # switch model text
        switch_model_text = wdg.QLabel()
        switch_model_text.setText("Select Model:")
        vbox1_utils.addWidget(switch_model_text, 1)
        
        # switch model combobutton
        self.combo_model = wdg.QComboBox(self)
        vbox1_utils.addWidget(self.combo_model, 1)
        self.model_combo_contents()
        self.model = os.path.join(ROOT_DIR, 'AI', 'models', self.combo_model.currentText() + '.caffemodel')
        vbox1_utils.addSpacing(6)
                
        # switch arch text
        switch_arch_text = wdg.QLabel()
        switch_arch_text.setText("Select Architecture:")
        vbox1_utils.addWidget(switch_arch_text, 1)
        
        # switch arch combobutton
        self.combo_arch = wdg.QComboBox(self)
        vbox1_utils.addWidget(self.combo_arch, 1)
        self.arch_combo_contents()
        self.architecture = os.path.join(ROOT_DIR, 'AI', 'models', self.combo_arch.currentText() + '.prototxt')
        vbox1_utils.addSpacing(10)
        
        # set model button
        model_btn = wdg.QPushButton('Set Model', self)
        model_btn.clicked.connect(self.switchmodelbutton)
        vbox1_utils.addWidget(model_btn, 1)
        vbox1_utils.addSpacing(40)
                
        # select cvg_threshold text
        cvg_threshold_text = wdg.QLabel()
        cvg_threshold_text.setText("Coverage threshold:")
        vbox1_utils.addWidget(cvg_threshold_text, 1)   
        
        # select cvg_threshold textfield
        self.cvg_threshold = wdg.QLineEdit(self)
        self.cvg_threshold.setText('0.6')
        vbox1_utils.addWidget(self.cvg_threshold, 1)
        vbox1_utils.addSpacing(5)
        
        # select rect_threshold text
        rect_threshold_text = wdg.QLabel()
        rect_threshold_text.setText("Rectangle threshold:")
        vbox1_utils.addWidget(rect_threshold_text, 1)   
        
        # select rect_threshold textfield
        self.rect_threshold = wdg.QLineEdit(self)
        self.rect_threshold.setText('3')
        vbox1_utils.addWidget(self.rect_threshold, 1)
        vbox1_utils.addSpacing(10)
        
        # set treshold button
        tresh_btn = wdg.QPushButton('Set threshold', self)
        tresh_btn.clicked.connect(self.settresholdbutton)
        vbox1_utils.addWidget(tresh_btn, 1)
        
        #spacer
        #verticalSpacer = wdg.QSpacerItem(40, 60, wdg.QSizePolicy.Minimum, wdg.QSizePolicy.Expanding)
        #vbox1_utils.addItem(verticalSpacer)
                
        vbox.addSpacing(80)
        
        # OUTPUT SECTION
        
        groupBox1_output = wdg.QGroupBox("OUTPUT", self)
        vbox.addWidget(groupBox1_output)
        vbox1_output = wdg.QVBoxLayout(groupBox1_output)
        vbox1_output.addSpacing(10)

        # GPU checkbox
        self.gpu_checkbox = wdg.QCheckBox('Use GPU', self)
        self.gpu_checkbox.stateChanged.connect(self.use_gpu_changed)
        #FMC_settings.useGPU = gpu_checkbox.checkState()
        vbox1_output.addWidget(self.gpu_checkbox, 1, Qt.AlignTop)
        vbox1_output.addSpacing(15)

        # predict button
        predict_btn = wdg.QPushButton('Predict', self)
        predict_btn.clicked.connect(self.predictbutton)
        vbox1_output.addWidget(predict_btn, 1)
        vbox1_output.addSpacing(15)
        
        # statistics button
        stat_btn = wdg.QPushButton('Create stats', self)
        stat_btn.clicked.connect(self.statisticsbutton)
        vbox1_output.addWidget(stat_btn, 1)
        vbox1_output.addSpacing(10)
        
        
        vbox.addSpacing(150)
               
        hbox.addWidget(groupBox1, 1)
        
        # menu group ends here        
# =============================================================================
        
        # image group starts here
        vbox2 = wdg.QHBoxLayout() # full Image Layout
        groupBox2 = wdg.QGroupBox("Image", self) # group box with the image
           
        vbox3 = wdg.QHBoxLayout(groupBox2)        
        vbox2.addWidget(groupBox2)
        hbox.addLayout(vbox2, 10)        
        
        self.vbox = vbox3 # to be able to switch the image        
        
        # defaul background
        background_image_path = os.path.join(ROOT_DIR, "util", "FMC.png")
        pixmap = set_image(self, background_image_path, mode='Single')
        switch_background(self, pixmap, mode='Start')
        
        # scrollbar
        self.scrollbar = wdg.QScrollBar()
        self.setSliderMax(1)
                        
        self.progress = wdg.QProgressBar()
        self.progress.setOrientation(Qt.Vertical)
        self.progress.setValue(0)

        vbox2.addWidget(self.scrollbar, 1)     
        vbox2.addWidget(self.progress, 1) 

        # image group ends here
# =============================================================================   
     
        self.setLayout(hbox)
        self.show()
        
    def loadbutton(self):
        self.global_image_path_list = []
        self.wheelValue = 0

        file_name = wdg.QFileDialog()
        names = file_name.getOpenFileNames(self, "Open files","","Images (*.jpg *.jpeg *.png *.tif *.tiff)")        
        
        load_images_from_dir(self, names[0])
        
        self.setSliderMax(len(self.global_image_path_list))

    def use_gpu_changed(self):
        if self.gpu_checkbox.checkState() == 0:
            FMC_use_gpu = False
        else:
            FMC_use_gpu = True
        
    def predictbutton(self):
        
        pm = set_image(self, self.global_image_path_list[self.wheelValue], mode='Single')
        switch_background(self, pm)
        
        pixmapToPredict = self.vbox.itemAt(0).widget().pixmap()        
        if (type(self.global_image_path_list) is list):
            print(FMC_use_gpu)
            print([self.global_image_path_list[self.wheelValue]])
            #rectangles = predict_one(self, [self.global_image_path_list[self.wheelValue]], FMC_settings.use_gpu)
            rectangles, colors, labels = predict_image_torch(opt_predict, [self.global_image_path_list[self.wheelValue]])
            print(rectangles, colors, labels)
            drawMulticlassRect(pixmapToPredict, rectangles, colors, labels)
            #pix = qg.QPixmap.fromImage(rectangles)

            pm = set_image(self, pixmapToPredict, mode='Pixmap')
            switch_background(self, pm)

        self.progress.setValue(0)
       
    def statisticsbutton(self):
        
        self.progress.setValue(0)
        
        rectangles = predict_all(self, self.global_image_path_list, FMC_use_gpu)
        
        with open('result.csv', 'wb') as csvfile:
            fieldnames = ['Image Name', 'Cell ID', 'x1', 'y1', 'x2', 'y2']
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)        
            writer.writeheader()
            
            cellID = 0
            imgID = 0
            for row in rectangles:
                if row[-1] != imgID:
                    imgID = row[-1]
                    cellID = 1
                else:
                    cellID += 1
                imgName = self.global_image_path_list[row[-1]].split(os.sep)
                
                writer.writerow({'Image Name': imgName[-1], 'Cell ID': cellID, 'x1': int(row[0]), 'y1': int(row[1]), 'x2': int(row[2]), 'y2': int(row[3])})
                
                
        self.progress.setValue(0)
        wdg.QMessageBox.about(self, "Prediction result", "The result file is placed in the graphics/ folder.")
        
    def settresholdbutton(self):
        change_params_in_deploy(self)              
        
    def switchmodelbutton(self):
        self.model = os.path.join(ROOT_DIR, 'AI', 'models', self.combo_model.currentText() + '.caffemodel')
        print(self.model)
        self.architecture = os.path.join(ROOT_DIR, 'AI', 'models', self.combo_arch.currentText() + '.prototxt')
        wdg.QMessageBox.about(self, "Model change", "The model files have been changed.")
      
    def model_combo_contents(self):
        dirpath = os.path.join(ROOT_DIR, 'AI', 'models')
        for filepath in os.listdir(dirpath):
            if filepath.endswith('.caffemodel'):
                fp = filepath.split('.')
                self.combo_model.addItem(fp[-2])
                
    def arch_combo_contents(self):
        dirpath = os.path.join(ROOT_DIR, 'AI', 'models')
        for filepath in os.listdir(dirpath):
            if filepath.endswith('.prototxt'):
                fp = filepath.split('.')
                self.combo_arch.addItem(fp[-2])
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if ((self.wheelValue - (delta and delta // abs(delta))) >= 0) and ((self.wheelValue - (delta and delta // abs(delta))) <= self.scrollbar.maximum()):
            self.wheelValue -= (delta and delta // abs(delta))
        self.scrollbar.setValue(self.wheelValue)
        print(self.wheelValue)
        
        if (type(self.global_image_path_list) is list) and len(self.global_image_path_list) > 0:
            self.global_image_path_list = scroll_image(self, self.global_image_path_list)

    def setSliderMax(self, maxValue):
        self.scrollbar.setMaximum(maxValue-1)

if __name__ == '__main__':
    
    app = wdg.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())