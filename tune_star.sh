#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

n_nodes=1
nproc=2

bs=1
accu_steps=32
NUM_FRAMES=4
MODEL_BASE=Efficient-Large-Model/VILA1.5-3B
# Final output checkpoint path

#DATA_SELECT="star_qa_imgx4_Program_Image"
#OUTPUT="./ckpts/star_qa_imgx4_Program_Image_vila3b_loss2"

#DATA_SELECT="star_qa_imgx4_Query_Image_Gen_Program"
#OUTPUT="./ckpts/star_qa_imgx4_Query_Image_Gen_Program_vila3b_loss2"

#DATA_SELECT="star_qa_imgx4_Query_Image"
#OUTPUT="./ckpts/star_qa_imgx4_Query_Image_vila3b_loss2"

####################################################################################
#DATA_SELECT="star_qa_Query_Program_Video"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Program_Video_vila3b_loss2"

#DATA_SELECT="star_qa_imgx4_Query_Program_Graph_Video"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Program_Graph_Video_vila3b_loss2"

#DATA_SELECT="star_qa_Query_Video_Gen_Program"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Video_Gen_Program_vila3b_loss2"

#DATA_SELECT="star_qa_Program_Video_Middle"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Program_Video_Middle_vila3b_loss2"

#DATA_SELECT="star_qa_Query_Video_Gen_Program_Middle"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Video_Gen_Program_Middle_vila3b_loss2"

DATA_SELECT="star_qa_Query_Video_Gen_Program_Middle_OptEarly"
OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Video_Gen_Program_Middle_OptEarly_vila3b_loss2"

torchrun --nnodes=$n_node --nproc_per_node=$nproc --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --model_max_length 4096 \
    --version v1 \
    --num_video_frames $NUM_FRAMES \
    --data_mixture $DATA_SELECT \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $accu_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb

#DATA_SELECT="star_qa_Query_Video"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Video_vila3b_loss2"

#DATA_SELECT="star_qa_Program_Video"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Program_Video_vila3b_loss2"

#DATA_SELECT="star_qa_imgx4_Query_Graph"
#OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Graph_vila3b_loss2"

DATA_SELECT="star_qa_Query_Video_Gen_Program_Middle_NoTab"
OUTPUT="./ckpts_vid/star_qa_imgx{$NUM_FRAMES}_Query_Video_Gen_Program_Middle_NoTab_vila3b_loss2"

torchrun --nnodes=$n_node --nproc_per_node=$nproc --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --model_max_length 4096 \
    --version v1 \
    --num_video_frames $NUM_FRAMES \
    --data_mixture $DATA_SELECT \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $accu_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb


