# vgg-fpn
vgg-fpn implemented by MXNet


## Generate fusion and quantization format model
```
python ocr_gen_qsym_mkldnn.py
```
## Run fusion+quantization
```
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=20
numactl --physcpubind=0-19 --membind=0 python inference.py --symbol-file=OCR-quantized-1batches-naive-symbol.json --param-file=OCR-quantized-0000.params --batch-size=16
```
