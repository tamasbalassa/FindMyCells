from __future__ import division

from models import *
from util.utils import *
from util.datasets import *
from util.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser_train = argparse.ArgumentParser()
parser_train.add_argument("--epochs", type=int, default=30, help="number of epochs")
# parser_train.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser_train.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser_train.add_argument("--model_config_path", type=str, default="AI/configs/yolov3.cfg", help="path to model config file")
parser_train.add_argument("--data_config_path", type=str, default="AI/configs/ilida_data.data", help="path to data config file")
# parser_train.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser_train.add_argument("--class_path", type=str, default="coco.names", help="path to class label file")
# parser_train.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
# parser_train.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser_train.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
# parser_train.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser_train.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser_train.add_argument("--checkpoint_dir", type=str, default="AI/models/checkpoints", help="directory where model checkpoints are saved")
parser_train.add_argument("--use_cuda", type=bool, default=False, help="whether to use cuda if available")
opt_train = parser_train.parse_args()
print(opt_train)

cuda = torch.cuda.is_available() and opt_train.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("AI/models/checkpoints", exist_ok=True)

classes = load_classes(opt_train.class_path)

# Get data configuration
data_config = parse_data_config(opt_train.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt_train.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt_train.model_config_path)
# model.load_weights(opt.weights_path)
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt_train.batch_size, shuffle=False, num_workers=opt_train.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt_train.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt_train.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % opt_train.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt_train.checkpoint_dir, epoch))
