#!/bin/bash
MODEL_PATH=ckpts_vid/kg_qa_imgx{4}_1_shot_vila3b_loss2_epoch100_lora_v2_Merged   
export CUDA_VISIBLE_DEVICES=1
CKPT=result_kg

python inference_test/evaluate_kg_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames 4 \
    --question-file ./dataset/kg-llm/rephrased_QA_25Oct24_v2.3/testing_SFT.json \
    --image-folder ./dataset/kg-llm/COIN/videos/ \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1 > TUNE_KG_test_imgx4_epoch100_1_shot_lora.txt


