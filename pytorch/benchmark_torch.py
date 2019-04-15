import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import pickle as cPickle
import logging
import argparse
import time
logging.basicConfig(level=logging.DEBUG)

from easydict import EasyDict as edict
cfg = edict()
cfg.side_vertex_pixel_threshold =  0.9
cfg.trunc_threshold = 0.1
cfg.epsilon = 1e-4

parser = argparse.ArgumentParser(description='Pytorch SymbolAPI-based vgg11FPN inference performance benchmark')
parser.add_argument('--batch-size', type=int, default=0,
                     help='Batch size to use for benchmarking. Example: 32, 64, 128.'
                          'By default, runs benchmark for batch sizes - 1, 32, 64, 128, 256')
opt = parser.parse_args()

def score(batch_size, num_batches):
    from vgg11FPN import East as East_net
    state_dict = torch.load('model/test_576.pth', map_location='cpu')['net']
    net = East_net(isTrain=False)
    net.load_state_dict({k:v for k,v in state_dict.items() if k in net.state_dict()})

    data_ = torch.randn(batch_size, 3, 1024, 1024)

    if torch.cuda.is_available():
        data_= data_.cuda()
        net.cuda()

    net.eval()

    # run
    dry_run = 5  # use 5 iterations to warm up
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        for i in range(dry_run + num_batches):
            if i == dry_run:
                tic = time.time()
            a, b = net(data_)
    prof.export_chrome_trace("./json/torch_cpu.json")
    # return num images per second
    return num_batches * batch_size / (time.time() - tic)


if __name__ == '__main__':
    if opt.batch_size == 0:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    else:
        batch_sizes = [opt.batch_size]

    dtype = "float32"
    logging.info('network: vgg11FPN')
    for b in batch_sizes:
        speed = score(batch_size=b, num_batches=10)
        logging.info('batch size %2d, dtype %s, images/sec: %f', b, dtype, speed)
