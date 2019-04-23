import mxnet as mx
import time
import logging
import numpy as np
from mxnet import Context
logging.basicConfig(level=logging.DEBUG)

SHAPE = (4, 3, 256, 256)
data = mx.symbol.Variable('data', shape=SHAPE)
sym = mx.symbol.UpSampling(data, scale=2, sample_type='nearest')

dry_run = 5
count = 100
arg_shapes, _, aux_shapes = sym.infer_shape()
arg_array = [mx.nd.random.uniform(-1, 1, shape=shape) for shape in arg_shapes]
aux_array = [mx.nd.random.uniform(shape=shape) for shape in aux_shapes]
exe = sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
for i in range(dry_run + count):
    if i == dry_run:
        tic = time.time()
    q = exe.forward(is_train=False)
    q[0].wait_to_read()
time_cost = time.time() - tic
logging.info('%s cycles time cost: %s', count, time_cost)