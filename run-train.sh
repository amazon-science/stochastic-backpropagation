
IMAGENET_DATA_PATH=~/imagenet/ # change it to your path

## training script for convnext_tiny_sbp05
DROP_PATH=0.1
LEARNING_RATE=4e-3
BATCH_SIZE=128
UPDATE_FREQ=4
MODEL_NAME=convnext_tiny_sbp05

SAVE_EXPS_DIR=convnext_tiny_sbp05_res # change it to your experiments directory
SEED=111

python -m torch.distributed.launch --nproc_per_node 8 main.py \
    --model $MODEL_NAME --drop_path $DROP_PATH \
    --batch_size $BATCH_SIZE --lr $LEARNING_RATE \
    --update_freq $UPDATE_FREQ \
    --data_path $IMAGENET_DATA_PATH \
    --use_amp true \
    --model_ema true --model_ema_eval true \
    --output_dir $SAVE_EXPS_DIR


## training script for convnext_base_sbp05 on 4 machines
DROP_PATH=0.3
LEARNING_RATE=4e-3
BATCH_SIZE=64
UPDATE_FREQ=2
MODEL_NAME=convnext_base_sbp05

SAVE_EXPS_DIR=convnext_base_sbp05_res # change it to your experiments directory
SEED=222

# change the node_rank on other nodes
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 4 --node_rank 0 \
    --master_addr="123.321.33.66"  --master_port 3366  main.py \
    --model $MODEL_NAME --drop_path $DROP_PATH \
    --batch_size $BATCH_SIZE --lr $LEARNING_RATE \
    --seed $SEED \
    --update_freq $UPDATE_FREQ \
    --data_path $IMAGENET_DATA_PATH \
    --use_amp true \
    --model_ema true --model_ema_eval true \
    --output_dir $SAVE_EXPS_DIR


## training script for vit_tiny_sbp05
DROP_PATH=0.0
LEARNING_RATE=2e-3
BATCH_SIZE=256
UPDATE_FREQ=1
MODEL_NAME=vit_tiny_sbp05

SAVE_EXPS_DIR=vit_tiny_sbp05_res # change it to your experiments directory
SEED=333

python -m torch.distributed.launch --nproc_per_node 8 main.py \
    --model $MODEL_NAME --drop_path $DROP_PATH \
    --batch_size $BATCH_SIZE --lr $LEARNING_RATE \
    --update_freq $UPDATE_FREQ \
    --data_path $IMAGENET_DATA_PATH \
    --use_amp true \
    --model_ema true --model_ema_eval true \
    --output_dir $SAVE_EXPS_DIR


## training script for vit_base_sbp05 on 4 machines
DROP_PATH=0.3
LEARNING_RATE=4e-3
BATCH_SIZE=64
UPDATE_FREQ=2
MODEL_NAME=vit_base_sbp05

SAVE_EXPS_DIR=vit_base_sbp05_res # change it to your experiments directory
SEED=444

# change the node_rank on other nodes
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 4 --node_rank 0 \
    --master_addr="123.321.33.66"  --master_port 3366  main.py \
    --model $MODEL_NAME --drop_path $DROP_PATH \
    --batch_size $BATCH_SIZE --lr $LEARNING_RATE \
    --seed $SEED \
    --update_freq $UPDATE_FREQ \
    --data_path $IMAGENET_DATA_PATH \
    --use_amp true \
    --model_ema true --model_ema_eval true \
    --output_dir $SAVE_EXPS_DIR
