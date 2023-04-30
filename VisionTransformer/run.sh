#!/bin/bash

# DeiT-small

python main.py --model deit_small_patch16_224 --data-path /home/datasets/imagenet/ --batch-size 512 --load deit_small_patch16_224-cd65a155.pth  --output_dir small_auto  --epochs 30 --warmup-epochs 0 --search-rank --distillation-type hard --teacher-model deit_small_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --with-align --distillation-without-token > small_auto.log

python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_small_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 256 --finetune-rank small_auto/checkpoint.pth  --output_dir small_auto_fintune --warmup-epochs 0 --distillation-type hard --teacher-model deit_small_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --with-align --distillation-without-token --warmup-epochs 0  --attn2-with-bias  --lr 1e-4 --min-lr 5e-6 --weight-decay 5e-3 > small_auto_finetune.log


# DeiT-Base
python main.py --model deit_base_patch16_224 --data-path /home/datasets/imagenet/ --batch-size 320 --resume-search-rank base_auto/checkpoint.pth --output_dir base_auto  --epochs 30 --warmup-epochs 0 --search-rank --distillation-type hard --teacher-model deit_base_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --with-align --distillation-without-token > base_auto.log
python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_base_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 256  --output_dir base_auto_finetune  --lr 1e-4 --min-lr 1e-6  --finetune-rank base_auto --unscale-lr --distillation-type hard --teacher-model deit_base_patch16_224 --teacher-path https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --distillation-without-token --warmup-epochs 0 --attn2-with-bias  > base_auto_finetune.log
