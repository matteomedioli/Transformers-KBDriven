source /data/medioli/env/bin/activate
export CUDA_VISIBLE_DEVICES=2 
export OMP_NUM_THREADS=32 
nohup python3 -m torch.distributed.launch --nproc_per_node 1 run_graph_sage.py &
