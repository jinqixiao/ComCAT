#!/bin/bash

# DeiT-small

wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

python main.py --model deit_small_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 512 --load deit_small_patch16_224-cd65a155.pth  --output_dir small_auto  --epochs 30 --warmup-epochs 0 --search-rank --distillation-type hard --teacher-model deit_small_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --with-align --distillation-without-token --batch-size-search 64 --target-params-reduction 0.5 > small_auto.log

python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_small_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 256 --finetune-rank-dir small_auto  --output_dir small_auto_finetune --warmup-epochs 0 --distillation-type hard --teacher-model deit_small_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --with-align --distillation-without-token --attn2-with-bias  --lr 1e-4 --min-lr 5e-6 --weight-decay 5e-3 > small_auto_finetune.log


# DeiT-Base

wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth

python main.py --model deit_base_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 320 --load deit_base_patch16_224-b5f2ef4d.pth --output_dir base_auto  --epochs 30 --warmup-epochs 0 --search-rank --distillation-type hard --teacher-model deit_base_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --with-align --distillation-without-token --batch-size-search 16 --target-params-reduction 0.6 > base_auto.log

python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_base_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 256  --output_dir base_auto_finetune  --lr 1e-4 --min-lr 1e-6  --finetune-rank-dir base_auto --unscale-lr --distillation-type hard --teacher-model deit_base_patch16_224 --teacher-path https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --distillation-without-token --warmup-epochs 0 --attn2-with-bias  > base_auto_finetune.log

# Test
mkdir small_79.58_0.44
curl -L -o small_79.58_0.44/ranks.txt "https://drive.google.com/uc?export=download&id=1kBuUelTCr7yBC4TQy6xn-HB_IfZwYSa7"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1GZmkwJzmCgrI9bAhxnAWgfKb8I_sD2mm" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1GZmkwJzmCgrI9bAhxnAWgfKb8I_sD2mm" -o small_79.58_0.44/checkpoint.pth
python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_small_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 256 --finetune-rank-dir small_79.58_0.44 --attn2-with-bias --eval

mkdir base_82.28_0.61
curl -L -o base_82.28_0.61/ranks.txt "https://drive.google.com/uc?export=download&id=1MDExPqwRSCwBelJrTjFDDxNjYKABXvf_"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1G3xDUZsXLkvv0ZCo5QmDKqCIsxgdBbnJ" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1G3xDUZsXLkvv0ZCo5QmDKqCIsxgdBbnJ" -o base_82.28_0.61/checkpoint.pth
python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_base_patch16_224 --data-path /raid/data/ilsvrc2012/ --batch-size 256 --finetune-rank-dir base_82.28_0.61 --attn2-with-bias --eval
