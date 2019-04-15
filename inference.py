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
import os
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import logging
import argparse
import time
import numpy as np
from importlib import import_module
from model import East

def onnx():
    # Import the ONNX model into MXNet's symbolic interface
    sym, arg, aux = onnx_mxnet.import_model("torch.onnx")
    print("Loaded torch_model.onnx!")
    print(sym.get_internals())
    sym.save("./model/torch.json")
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu(0)) for k, v in arg.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu(0)) for k, v in aux.items()})
    mx.nd.save("./model/torch.params", save_dict)

def load_model(symbol_file, param_file, logger=None):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if logger is not None:
        logger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params

def benchmark_score(symbol_file, ctx, batch_size, num_batches, data_layer_type, logger=None):
    # get mod
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    sym = mx.sym.load(symbol_file_path)
    mod = mx.mod.Module(symbol=sym, context=ctx)
    if data_layer_type == "int8":
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.int8)
    elif data_layer_type == 'uint8':
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.uint8)
    else:  # float32
        dshape = mx.io.DataDesc(name='data', shape=(
            batch_size,) + data_shape, dtype=np.float32)
    mod.bind(for_training=False,
             inputs_need_grad=False,
             data_shapes=[dshape])
    mod.init_params()

    # get data
    if data_layer_type == "float32":
        data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx, dtype=data_layer_type)
                for _, shape in mod.data_shapes]
    else:
        data = [mx.nd.full(shape=shape, val=127, ctx=ctx, dtype=data_layer_type)
                for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])  # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='cpu')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--image-shape', type=str, default='3,1024,1024')
    # parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, default=10, help='number of images used for inference')
    parser.add_argument('--data-layer-type', type=str, default="float32",
                        choices=['float32', 'int8', 'uint8'],
                        help='data type for data layer')

    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file

    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    data_layer_type = args.data_layer_type
    logger.info('Running model %s for inference' % symbol_file)
    speed = benchmark_score(symbol_file, ctx, batch_size, args.num_inference_batches, data_layer_type, logger)
    logger.info('batch size %2d, image/sec: %f', batch_size, speed)