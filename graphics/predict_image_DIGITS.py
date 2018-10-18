# this is a modified version of the file that was originally implemented by: lukeyeager
# source: https://gist.github.com/lukeyeager/777087991419d98700054cade2f755e6

import sys
import os
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import numpy as np
np.set_printoptions(threshold=np.nan)
import caffe
from caffe.proto import caffe_pb2
import time
from google.protobuf import text_format
import PIL.Image
import scipy.misc
import FMC_settings
import PyQt5.QtWidgets as wdg

def predict_one(self, img, gpu):

    #caffe.set_device(0)   

    caffemodel = str(self.model)
    deploy_file = str(self.architecture)
    
    net = get_net(self, caffemodel, deploy_file, gpu)
    
    bounding_boxes = classify(self, net, deploy_file, img)
    
    return bounding_boxes

def predict_all(self, imglist, gpu):
    
    caffemodel = str(self.model)
    deploy_file = str(self.architecture)
    net = get_net(self, caffemodel, deploy_file, gpu)
    
    bounding_boxes = classify(self, net, deploy_file, imglist)
    
    return bounding_boxes

def forward_pass(self, images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)
            
    dims = transformer.inputs['data'][1:]
    
    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in range(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        
        output = net.forward()[net.outputs[0]]
                
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        
        self.progress.setValue(float(len(scores))/len(caffe_images)*100)
        wdg.QApplication.processEvents()
        print('Processed %s/%s images in %f seconds ...', (len(scores), len(caffe_images), (end - start)))

    return scores

def get_transformer(deploy_file):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))
    
    return t

def classify(self, net, deploy_file, image_files,
        mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    
    transformer = get_transformer(deploy_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    
        
    images = [load_image(image_file, height, width, mode) for image_file in image_files] #- this works if we have multiple images

    FMC_settings.gInputWidth = width
    FMC_settings.gInputHeight = height
    
    # Classify the image    
    scores = forward_pass(self, images, net, transformer, batch_size=batch_size)

    ### Process the results
    rects = []
    # Format of scores is [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    # https://github.com/NVIDIA/caffe/blob/v0.15.13/python/caffe/layers/detectnet/clustering.py#L81
    for i, image_results in enumerate(scores):
        for left, top, right, bottom, confidence in image_results:
            if confidence == 0:
                continue
            
            if left<0:left=0
            if top<0:top=0
            if right<0:right=0
            if bottom<0:bottom=0
            
            rects.append([left, top, right, bottom, i])
                 
    return rects


def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def get_net(self, caffemodel, deploy_file, use_gpu=False):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
           
    return net