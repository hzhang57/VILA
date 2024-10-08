#!/bin/bash
MODEL_PATH=Efficient-Large-Model/VILA1.5-3B
CKPT=result_kg

python inference_test/evaluate_star.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 8 \
    --question-file ./dataset/star/sft_annots/STAR_val_NEAT_imgx4_Program_Graph_v3.0.json \
    --image-folder ./dataset/star/charadesv1_480/frames \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1

