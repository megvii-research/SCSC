#rlaunch --positive-tags 2080ti --cpu=16 --gpu=8 --memory=200000 -P0  -- python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
[ ! -d ./log ] && mkdir ./log
rlaunch --max-wait-time=24h --preemptible=no --group=research_model_v100 --cpu=32 --gpu=8 --memory=120000  --  python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py | tee -a log/training.log

#python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py | tee -a log/training.log
