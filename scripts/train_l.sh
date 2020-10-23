#!/bin/bash
#
#SBATCH --job-name=ldetr
#SBATCH --output=logs/ldetr_%j.txt  # output file
#SBATCH -e logs/ldetr_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00       # Runtime in D-HH:MM
#SBATCH --gres=gpu:4
#SBATCH --mem 60G

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --ltransformer \
	--epochs 40 \
    --lr 1e-4 \
    --backbone resnet50 \
    --batch_size 1 \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/ldetr
	# --resume /mnt/nfs/scratch1/zhiyilai/detrs/cmdetr/checkpoint.pth \
    # --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
