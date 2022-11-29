# CUB
python main.py --seed 0 --dataset cub --imgsize 550 --crop 448 --model resnet50 --epochs 300 --batchsize 16 --gpu_ids 0,1

# aircraft
python main.py --seed 0 --dataset aircraft --imgsize 550 --crop 448 --model resnet50 --epochs 300 --batchsize 16 --gpu_ids 5,6

# car
python main.py --seed 0 --dataset car --imgsize 550 --crop 448 --model resnet50 --epochs 300 --batchsize 16 --gpu_ids 3,4

