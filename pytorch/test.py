import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import pickle as cPickle

from easydict import EasyDict as edict
cfg = edict()
cfg.side_vertex_pixel_threshold =  0.9
cfg.trunc_threshold = 0.1
cfg.epsilon = 1e-4


if __name__ == '__main__':

    from vgg11FPN import East as East_net
        #checkpoint = torch.load(args.checkpoint, map_location='cpu')
        #net.load_state_dict(checkpoint['net'])
    state_dict = torch.load('model/test_576.pth', map_location='cpu')['net']
    net = East_net(isTrain=False)
    # net.load_state_dict({k:v for k,v in state_dict.items() if k in net.state_dict()})

    net.cuda()
    net.eval()

    dummpy_input = torch.randn(1, 3, 1024, 1024, device='cuda')
    input_names = ["data", "softmax_label"]
    input_names += ["%s_weight" % i for i in range(105)]
    # input_names = ["data"]
    output_names = ["cls_preds", "link_preds"]
    torch.onnx.export(net, dummpy_input, 'test.onnx', verbose=True, input_names=input_names, output_names=output_names)
