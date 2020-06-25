#!/bin/bash
#
#SBATCH --job-name=comp_detr
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=00-12:00       # Runtime in D-HH:MM
#SBATCH --gres=gpu:1

# python run_with_submitit.py --ngpus 1 --nodes 1 --timeout 1 --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /mnt/nfs/scratch1/zhiyilai/coco --output_dir /mnt/nfs/scratch1/zhiyilai/comp/detr

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /mnt/nfs/scratch1/zhiyilai/coco --output_dir /mnt/nfs/scratch1/zhiyilai/comp/detr
