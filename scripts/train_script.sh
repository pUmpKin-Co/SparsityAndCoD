config=$1
batch_size=$2
global_train_batch_size=$3
learning_rate=$4
visible_gpus=$5
master_port=${6:-29500}
run_suffix=${7:-"none"}

export CUDA_VISIBLE_DEVICES=$visible_gpus
export WANDB_MODE=online
num_gpus=$(echo $visible_gpus | tr ',' '\n' | wc -l)


echo "Number of GPUs: $num_gpus"

torchrun --nproc-per-node=$num_gpus --master-port=$master_port scripts/train_torchrun.py \
    configs/$config.yaml \
    --run_name=$config-$run_suffix \
    --model.flash_attention=true \
    --model.layer_norm_with_affine=true \
    --model.max_sequence_length=1024 \
    --model.layer_norm_scaling=false \
    --optimizer.name="adamw" \
    --optimizer.learning_rate=$learning_rate \
    --optimizer.weight_decay=0.1 \
    --optimizer.eps=1e-8 \
    --optimizer.no_decay_norm_and_bias=true \
    --optimizer.metrics_log_interval=1 \
    --optimizer.record_update_metrics=false \
    --optimizer.spike_detection=false \
    --scheduler.t_warmup=1000 \
    --data.dir=$data_dir \
    --data.num_workers=8 \
    --data.pin_memory=true \
    --eval_interval=3000 \
    --tokenizer.identifier=$tokenizer_id \
    --save_folder="./runs/$config-$run_suffix" \
    --save_interval=5000 \
    --save_overwrite=true \
    --no_pre_train_checkpoint=true \
    --max_duration=5e9T \
    --global_train_batch_size=$global_train_batch_size \
    --device_train_microbatch_size=$batch_size \
    --max_grad_norm=1.0 \
    --precision=amp_bf16 \
    --wandb.project="StableLM" \
    --wandb.group="VARIANCE-MoE" \
    --wandb.name=$run_name \
    --wandb.log_interval=1 \
    --layerwise_statis_collect_interval=1