graph_size=35
dataset="update_config_35"
desc="lr1e-5_epoch1"
mode="without_qos"

mkdir -p ./result/$mode/$desc

accelerate launch \
    --gpu_ids $1 \
    --config_file ./accelerate_config/my_config_0.yaml train_graph_llm.py \
    --batch_size 1 \
    --num_epochs 0 \
    --adapter_dim 128 \
    --adapter_n_heads 1 \
    --n_encoder_layers 1 \
    --n_decoder_layers 1 \
    --lr 1e-5 \
    --dataset net \
    --dataset_path /home/jxh/jzt/Config/dataset_temp/$mode/$dataset \
    --save_path ./result/$mode/$desc/$dataset.$desc.pth \
    > ./result/$mode/$desc/$dataset.$desc.train.log