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

import argparse
import os
import logging
import pickle
from mxnet import nd
import mxnet as mx
from model import East
from mxnet.contrib.quantization import quantize_model
import ctypes

def download_calib_dataset(dataset_url, calib_dataset, logger=None):
    if logger is not None:
        logger.info('Downloading calibration dataset from %s to %s' % (dataset_url, calib_dataset))
    mx.test_utils.download(dataset_url, calib_dataset)

def load_model(symbol_file, param_file, mlogger=None):
    """load existing symbol model"""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if mlogger is not None:
        mlogger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if mlogger is not None:
        mlogger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    marg_params = {}
    maux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            marg_params[name] = v
        if tp == 'aux':
            maux_params[name] = v
    return symbol, marg_params, maux_params

def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gescription a calibrated quantized from a FP32 model")
    parser.add_argument('--ctx', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-shape', type=str, default='3,1024,1024')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--calib-dataset', type=str, default='data/val_256_q90.rec',
                        help='path of the calibration dataset')
    parser.add_argument('--num-calib-batches', type=int, default=1,
                        help='number of batches for calibration')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--calib-mode', type=str, default='naive',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='uint8',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    args = parser.parse_args()
    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx % is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    # get batch size
    batch_size = args.batch_size
    logger.info('batsh size = %d for calibration', batch_size)
    # get image shape
    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('input data shape = %s' % str(data_shape))
    data_nthreads = args.data_nthreads
    logger.info('shuffle_dataset=%s' % args.shuffle_dataset)
    # get number of batches and mode for calibratoin
    num_calib_batches = args.num_calib_batches
    calib_mode = args.calib_mode
    logger.info('calibation mode set to %s' % calib_mode)
    # download calibration dataset
    if calib_mode != 'none':
        download_calib_dataset('http://data.mxnet.io/data/val_256_q90.rec', args.calib_dataset)

    calib_layer = lambda name: name.endswith('_output') or name == 'data'
    if args.quantized_dtype == 'uint8':
        logger.info('quantized dtype is set to uint8, will exclude first conv')
    excluded_sym_names = []

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    sym, arg_params, aux_params = sym, arg_params, aux_params = load_model('torch.json', 'torch.params', logger)
    sym = sym.get_backend_symbol('MKLDNN')

    logger.info('Creating ImageRecordIter for reading calibration dataset')
    data = mx.io.ImageRecordIter(path_imgrec=args.calib_dataset,
                                 label_width=1,
                                 preprocess_threads=data_nthreads,
                                 batch_size=batch_size,
                                 data_shape=data_shape,
                                 label_name=label_name,
                                 rand_crop=False,
                                 rand_mirror=False,
                                 shuffle=args.shuffle_dataset,
                                 shuffle_chunk_seed=args.shuffle_chunk_seed,
                                 seed=args.shuffle_seed)

    qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                   ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                   calib_mode=calib_mode, calib_data=data,
                                                   num_calib_examples=num_calib_batches * batch_size,
                                                   calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                   label_names=(label_name,), logger=logger)

    if calib_mode == 'entropy':
        suffix = '-quantized-%dbatches-entropy' % num_calib_batches
    elif calib_mode == 'naive':
        suffix = '-quantized-%dbatches-naive' % num_calib_batches
    else:
        raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                         % calib_mode)
    prefix = 'OCR'
    sym_name = '%s-symbol.json' % (prefix + suffix)
    cqsym = qsym.get_backend_symbol('MKLDNN_POST_QUANTIZE')
    save_symbol(sym_name, cqsym, logger)
    param_name = '%s-%04d.params' % (prefix + '-quantized', 0)
    save_params(param_name, qarg_params, aux_params, logger)