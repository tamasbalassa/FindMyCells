import numpy as np
import caffe
import cv2

def predict_one(img,gpu):

    caffe.set_device(0)
    if gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()    
    
    net = caffe.Net('../AI/deploy.prototxt', '../AI/snapshot_iter_53928.caffemodel', caffe.TEST)
    
    #image = cv2.imread('../AI/tissue002a.jpg')
    image = cv2.imread(img)
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
    image = caffe.io.resize( image, (512,704))
    
    net.blobs['data'].reshape(1,image.shape[2], image.shape[0], image.shape[1])
    
    transformer = caffe.io.Transformer({'data': (1, image.shape[2], image.shape[0], image.shape[1])})
    transformer.set_transpose('data', (2,0,1))
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    
    xx = net.forward()[net.outputs[-1]]
    scores = np.copy(xx)
    
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
    
    
    out = net.blobs['coverage'].data
    coverage = out[0,0,:,:]
    
    #return coverage
    return scores