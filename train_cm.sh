#!/bin/bash
#
#SBATCH --job-name=train_roi
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=2080ti-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00       # Runtime in D-HH:MM
#SBATCH --gres=gpu:8
#SBATCH --mem 60G

python -m torch.distributed.launch --nproc_per_node=6 --use_env main.py --ctransformer --merge \
	--start_epoch 20 \
	--epochs 80 \
	--lr_drop 400 \
    # --lr 8e-6 \
	--resume /mnt/nfs/scratch1/zhiyilai/detrs/cmdetr/checkpoint.pth \
    --backbone resnet50 \
    --batch_size 1 \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/cmdetr
    # --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \

# python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --ctransformer --merge \
# 	--epochs 20 \
#     --lr 1e-5 \
#     --backbone resnet50 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
#     --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/cmdetr_small
