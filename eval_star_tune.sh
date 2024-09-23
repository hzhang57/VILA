#!/bin/bash
MODEL_PATH=Efficient-Large-Model/VILA1.5-3B
MODEL_PATH=ckpts/star_imgx4_vila3b
MODEL_PATH=ckpts/star_imgx4_v3_imgx4_VILA1.5_3B_20480_linear_Bank0_LearnLat0
MODEL_PATH=ckpts/star_imgx4_vila3b_loss2
CKPT=result_kg

python inference_test/evaluate_star.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 4 \
    --question-file ./dataset/star/sft_annots/STAR_val_NEAT_imgx4_3.0.json \
    --image-folder ./dataset/star/charadesv1_480/frames \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1

