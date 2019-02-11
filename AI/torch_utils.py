import datetime
import random

from models import Darknet
from util.datasets import ImageFolder
from util.utils import *
from util import FMC_settings


import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from PIL import Image
from PIL.ImageQt import ImageQt
from matplotlib.ticker import NullLocator



def predict_image_torch(opt_predict, imgpath):
    print(opt_predict)
    cuda = torch.cuda.is_available() and opt_predict.use_cuda

    # Set up model
    model = Darknet(opt_predict.config_path, img_size=opt_predict.img_size)
    model.load_weights(opt_predict.weights_path)

    if cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    dataloader = Image.open(imgpath[0])
    #FMC_settings.gInputWidth, FMC_settings.gInputHeight = dataloader.size
    #print(FMC_settings.gInputWidth, FMC_settings.gInputHeight)
    dataloader = dataloader.resize((opt_predict.img_size, opt_predict.img_size), Image.ANTIALIAS)
    print(dataloader)

    # TODO WHAT TO DO HERE?!?!
    classes = load_classes(opt_predict.class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    prev_time = time.time()
    ##############################################################

    # Configure input
    transform =ToTensor()
    t = transform(dataloader)
    input_imgs = Variable(t)
    input_imgs = input_imgs.unsqueeze(0)

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, opt_predict.conf_thres, opt_predict.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print('\t+ Inference Time: %s' % (inference_time))

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Create plot
    img = Image.open(imgpath[0])
    img = img.resize((opt_predict.img_size, opt_predict.img_size), Image.ANTIALIAS)
    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1)
    #plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt_predict.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt_predict.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt_predict.img_size - pad_y
    unpad_w = opt_predict.img_size - pad_x

    FMC_settings.gInputWidth = unpad_w
    FMC_settings.gInputHeight = unpad_h

    # Draw bounding boxes and labels of detections
    bboxes = []
    bbcolors = []
    bblabels = []
    if detections is not None:
        print(detections[0])
        print(detections[0][:, -1])
        unique_labels = detections[0][:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)

            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            bbcolors.append(color)
            bblabels.append(classes[int(cls_pred)])

    # plt.axis('off')
    # plt.tight_layout()
    # plt.gca().set_axis_off()
    # plt.gca().set_frame_on(False)
    # plt.gca().xaxis.set_major_locator(NullLocator())
    # plt.gca().yaxis.set_major_locator(NullLocator())
    #
    # # draw the renderer
    # fig.canvas.draw()
    #
    # # Get the RGBA buffer from the figure
    # w, h = fig.canvas.get_width_height()
    # print(w, h)
    # buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    # buf.shape = (w, h, 4)
    #
    # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)
    #
    # w, h, d = buf.shape
    # print(w, h)
    #
    # im = Image.frombytes("RGBA", (w, h), buf.tostring())
    # #im = im.resize((opt_predict.img_size, opt_predict.img_size), Image.ANTIALIAS)
    # qim = ImageQt(im)
    #
    # im.show()
    #
    # return qim

    return bboxes, bbcolors, bblabels
