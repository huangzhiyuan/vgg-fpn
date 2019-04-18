# vgg-fpn
vgg-fpn implemented by MXNet

## Before optimization
```
# For GPU
python benchmark.py

# For CPU
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=CORES_PER_SOCKET
python benchmark.py --dev cpu
```

## After optimization
### 1. Generate fusion and quantization format model
```
python ocr_gen_qsym_mkldnn.py
```
### 2. Run fusion+quantization
```
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=CORES_PER_SOCKET
python inference.py --symbol-file=OCR-quantized-1batches-naive-symbol.json --param-file=OCR-quantized-0000.params --batch-size=16
```
