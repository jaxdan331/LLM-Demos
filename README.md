# 大模型多卡推理

一个简单的脚本，使用Pytorch实现大模型的多卡（并行、分布式）推理

## 并行推理
```
# 以 8 张显卡为例
torchrun --standalone --nnodes=1 --nproc_per_node=8 multi_gpus_inference.py \
         --model_path <model_path> \
         --data_path <data_path> \
         --save_path <save_path> \
         --batch_size <batch_size>
```

## 分布式推理
```
torchrun --standalone --nnodes=1 --nproc_per_node=1 multi_gpus_inference.py \
         --model_path <model_path> \
         --data_path <data_path> \
         --save_path <save_path> \
         --batch_size <batch_size>
```
