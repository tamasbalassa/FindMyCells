import numpy as np
np.set_printoptions(threshold=np.nan)
import caffe
from caffe.proto import caffe_pb2
import time
from google.protobuf import text_format
import PIL.Image
import scipy.misc
from skimage.measure import label, regionprops
from scipy.misc import imresize

def predict_one(img,gpu):

    caffe.set_device(0)
    if gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()    
    
    
    caffemodel = '../AI/snapshot_iter_53928.caffemodel'
    deploy_file = '../AI/deploy_krisztian.prototxt'
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    #net = get_net(caffemodel, deploy_file, gpu)
    
    #img = ['tissue002a.jpg','tissue3.jpg', '20.jpg', '199.jpg']
    #img = ['tissue002a.jpg']
    
    bounding_boxes, regprops = classify(net, deploy_file, img)
    
    return bounding_boxes, regprops


def forward_pass(images, net, transformer, batch_size=None):
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
    regprops = []
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[0]]
        
        
        y = net.blobs['coverage'].data[0,0]
        y = imresize(y, [850, 1200], mode='F')
        low_values_flags = y < 0.001
        high_values_flags = y >= 0.001
        y[low_values_flags] = 0
        y[high_values_flags] = 1
        y_int = y.astype(int)
        label_img = label(y_int)
        x = regionprops(label_img)
        
        #x_multiplier = net.blobs['data'].data.shape[3] / net.blobs['coverage'].data[0,0].shape[1]
        #y_multiplier = net.blobs['data'].data.shape[2] / net.blobs['coverage'].data[0,0].shape[0]
        #print x_multiplier
        #print y_multiplier
        if len(x) > 0:
            for one_rect in x:
                y0, x0 = one_rect.centroid
                #y1 = y0 * y_multiplier
                #x1 = x0 * x_multiplier
                regprops.append([x0,y0])
                #print y0, x0
            
# =============================================================================
#         y1 = net.blobs['bboxes'].data
#         print y1.shape
# =============================================================================
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    print scores
    return scores, regprops

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

def classify(net, deploy_file, image_files,
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
    # Load the model and images
    #net = get_net(caffemodel, deploy_file, use_gpu)
    
    transformer = get_transformer(deploy_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    
    images = [load_image(image_file, height, width, mode) for image_file in image_files] #- this works if we have multiple images
    #images = load_image(image_files, height, width,mode)
    #labels = read_labels(labels_file)

    # Classify the image
    scores, regprops = forward_pass(images, net, transformer, batch_size=batch_size)

    ### Process the results
    rects = []
    # Format of scores is [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    # https://github.com/NVIDIA/caffe/blob/v0.15.13/python/caffe/layers/detectnet/clustering.py#L81
    for i, image_results in enumerate(scores):
        print '==> Image #%d' % i
        for left, top, right, bottom, confidence in image_results:
            if confidence == 0:
                continue

            print 'Detected object at [(%d, %d), (%d, %d)] with "confidence" %f' % (
                int(round(left)),
                int(round(top)),
                int(round(right)),
                int(round(bottom)),
                confidence,
            )
            
            rects.append([left, top, right, bottom])
                 
    return rects, regprops


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
    print path
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def get_net(caffemodel, deploy_file, use_gpu=True):
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

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)