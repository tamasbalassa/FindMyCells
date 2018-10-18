import PyQt5.QtWidgets as wdg
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

# this function is called when the "Set treshold" is pressed
# it modifies two parameters in the deploy (prototxt) file
def change_params_in_deploy(self):
    
    nnet = caffe_pb2.NetParameter()    
    deploy_file = '../AI/models/' + self.combo_arch.currentText() + '.prototxt'
    
    with open(deploy_file) as f:
        s = f.read()
        txtf.Merge(s, nnet)
        
    layerNames = [l.name for l in nnet.layer]
    idx = layerNames.index('cluster')
    l = nnet.layer[idx]
    
    params = l.python_param.param_str
    params = params.split(',')
    params[3] = float(self.cvg_threshold.text())
    if params[3]<0:params[3]=0.6
    params[4] = int(self.rect_threshold.text())
    if params[4]<0:params[4]=3
    updated_param_str = ','.join(str(e) for e in params)
    l.python_param.param_str = updated_param_str
    
    print('writing', deploy_file)
    with open(deploy_file, 'w') as f:
        f.write(str(nnet))

    wdg.QMessageBox.about(self, "Parameter change", "The model parameter has been changed.")
# =============================================================================
#     3. gridbox_cvg_threshold
#     4. gridbox_rect_threshold
#     5. gridbox_rect_eps
#     6. min_height
# =============================================================================