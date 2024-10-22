#!/bin/bash
MODEL_PATH=Efficient-Large-Model/VILA1.5-3B
CKPT=result_kg

python inference_test/evaluate_star_m_round_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 4 \
    --question-file ./dataset/star/sft_annots_video_v3.3/STAR_val_NEAT_Query_Gen_Program_Video_v3.3.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1 > ZeroShot_STAR_val_NEAT_Query_Gen_Program_Video_v3.3.txt

