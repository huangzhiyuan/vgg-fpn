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
import mxnet as mx
import numpy as np

def MyAdaptiveAvgPool2d(x, **kwargs):
    # inp_size = x.size()
    # return mx.symbol.Pooling(data=x, pool_type="avg", kernel=(inp_size[2], inp_size[3]))
    return mx.symbol.Pooling(data=x, pool_type="avg", kernel=(256,256))

def convolution(x, k, inp_dim, out_dim, stride=1, with_bn=True, **kwargs):
    pad = (k - 1) // 2
    conv = mx.symbol.Convolution(x, num_filter=out_dim, kernel=(k,k), pad=(pad,pad), stride=(stride,stride), bias=not with_bn)
    bn = mx.symbol.BatchNorm(conv)
    relu = mx.symbol.Activation(bn, act_type='relu')
    return relu

def make_layers(x, in_channels, channels_list, batch_norm=True):
    for v in channels_list:
        conv = mx.symbol.Convolution(x, num_filter=v, kernel=(3,3), pad=(1,1))
        if batch_norm:
            bn = mx.symbol.BatchNorm(conv)
            relu = mx.symbol.Activation(bn, act_type='relu')
        else:
            relu = mx.symbol.Activation(conv, act_type='relu')
        x = relu
    return relu

def VGG11bn(x, init_weights=True):
    blocks = []
    x = make_layers(x, 3, [64])
    x = mx.symbol.Pooling(x, pool_type="max", kernel=(3,3), stride=(2,2), pad=(1,1,1,1))  #2

    x = make_layers(x, 64, [128])
    x = mx.symbol.Pooling(x, pool_type="max", kernel=(3,3), stride=(2,2), pad=(1,1,1,1))  #4
    blocks.append(x)

    x = make_layers(x, 128, [256, 256])
    x = mx.symbol.Pooling(x, pool_type="max", kernel=(3,3), stride=(2,2), pad=(1,1,1,1))  #8
    blocks.append(x)

    x = make_layers(x, 256, [512, 512])
    x = mx.symbol.Pooling(x, pool_type="max", kernel=(3,3), stride=(2,2), pad=(1,1,1,1))  #16
    blocks.append(x)

    x = make_layers(x, 512, [512, 512])
    x = mx.symbol.Pooling(x, pool_type="max", kernel=(3,3), stride=(2,2), pad=(1,1,1,1))  #32
    blocks.append(x)

    return x, blocks

def _make_head(x, out_planes):
    for _ in range(3):
        conv = mx.symbol.Convolution(x, num_filter=32, kernel=(3,3), pad=(1,1), stride=(1,1))
        relu = mx.symbol.Activation(conv, act_type='relu')
        x = relu
    conv = mx.symbol.Convolution(relu, num_filter=out_planes, kernel=(3,3), pad=(1,1), stride=(1,1))
    return conv

def East(feat_stride=4, isTrain=True):
    input = mx.sym.var('data')
    input = mx.sym.Variable('data')
    _, f = VGG11bn(input)
    # bs 2048 w/32 h/32
    h = f[3]
    # bs 2048 w/16 h/16
    g = mx.symbol.UpSampling(h, scale=2, sample_type='nearest')
    c = mx.symbol.concat(g, f[2])
    c = mx.symbol.Convolution(c, num_filter=128, kernel=(1,1), stride=(1,1))  # ic 512+512
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')

    # bs 128 w/16 h/16
    c = mx.symbol.Convolution(r, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1))
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')


    # bs 128 w/8 h/8
    g = mx.symbol.UpSampling(r, scale=2, sample_type='nearest')
    c = mx.symbol.concat(g, f[1])
    c = mx.symbol.Convolution(c, num_filter=64, kernel=(1,1), stride=(1,1))  # ic 128+256
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')

    # bs 64 w/8 h/8
    c = mx.symbol.Convolution(r, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1))
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')

    # bs 64 w/4 h/4
    g = mx.symbol.UpSampling(r, scale=2, sample_type='nearest')
    c = mx.symbol.concat(g, f[0])
    c = mx.symbol.Convolution(c, num_filter=64, kernel=(1,1), stride=(1,1))  # ic 64+128
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')

    # bs 32 w/4 h/4
    c = mx.symbol.Convolution(r, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1))
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')
    c = mx.symbol.Convolution(r, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1))
    b = mx.symbol.BatchNorm(c)
    r = mx.symbol.Activation(b, act_type='relu')

    cls_preds = _make_head(r, 2)
    link_preds = _make_head(r, 16)
    cls_preds = MyAdaptiveAvgPool2d(cls_preds)

    return cls_preds, link_preds
