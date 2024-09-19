#!/bin/bash
MODEL_PATH=Efficient-Large-Model/VILA1.5-3B
CKPT=result_kg

python inference_test/evaluate_kg_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 6 \
    --question-file ./dataset_kg/QA_12Sept24/test_vqa.json \
    --image-folder ./dataset_kg/QA_12Sept24/COIN/videos/ \
    --answers-file ./dataset_kg/QA_12Sept24/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

