#!/bin/bash
#
#SBATCH --job-name=train_roi
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00       # Runtime in D-HH:MM
#SBATCH --gres=gpu:4
#SBATCH --mem 60G

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --ctransformer --merge \
	--start_epoch 5\
	--epochs 10 \
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume /mnt/nfs/scratch1/zhiyilai/detrs/cmdetr/checkpoint.pth \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/cmdetr
