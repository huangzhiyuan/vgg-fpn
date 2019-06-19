# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Benchmark the scoring performance on vgg11FPN
"""
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import logging
import argparse
import time
import numpy as np
from importlib import import_module
from model import East
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='SymbolAPI-based vgg11FPN inference performance benchmark')
parser.add_argument('--batch-size', type=int, default=1,
                     help='Batch size to use for benchmarking. Example: 32, 64, 128.'
                          'By default, runs benchmark for batch sizes - 1, 32, 64, 128, 256')
parser.add_argument('--dev', type=str, default='cpu')
parser.add_argument('--latency', type=bool, default=True)
opt = parser.parse_args()

def onnx():
    # Import the ONNX model into MXNet's symbolic interface
    sym, arg, aux = onnx_mxnet.import_model("test.onnx")
    print("Loaded torch_model.onnx!")
    print(sym.get_internals())
    sym.save("./model/torch.json")
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu(0)) for k, v in arg.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu(0)) for k, v in aux.items()})
    mx.nd.save("./model/torch.params", save_dict)

def score(dev, latency, batch_size, num_batches):
    sym1, sym2 = East(isTrain=False)
    sym = mx.sym.Group([sym1, sym2])

    if 'cpu' in str(dev):
       sym = sym.get_backend_symbol('MKLDNN')
    # sym, arg, aux = onnx_mxnet.import_model("test.onnx")

    data_shape = [('data', (batch_size, 3, 1024, 1024))]
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad=False,
             data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    if latency:
        logging.info('latency: %f ms', (time.time() - tic) / num_batches * 1000)
    # return num images per second
    return num_batches * batch_size / (time.time() - tic)


if __name__ == '__main__':
    if opt.batch_size == 0:
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    else:
        batch_sizes = [opt.batch_size]

    devs = []
    if opt.dev == 'cpu':
        devs = [mx.cpu()]
    else:
        devs = [mx.gpu(0)]

    dtype = "float32"
    logging.info('network: vgg11FPN')
    for d in devs:
        logging.info('device: %s', d)
        for b in batch_sizes:
            speed = score(dev=d, latency=opt.latency, batch_size=b, num_batches=50)
            logging.info('batch size %2d, dtype %s, images/sec: %f', b, dtype, speed)
