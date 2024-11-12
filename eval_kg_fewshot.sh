#!/bin/bash

#MODEL_PATH=Efficient-Large-Model/VILA1.5-3B-S2
MODEL_PATH=Efficient-Large-Model/Llama-3-LongVILA-8B-128frames
export CUDA_VISIBLE_DEVICES=0
CKPT=result_kg

python inference_test/evaluate_kg_video_fewshot.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 20 \
    --num_video_frames 4 \
    --question-file ./dataset/kg-llm/rephrased_QA_25Oct24_v2.3/testing_SFT.json \
    --N_shot 5 \
    --few_shot_file ./dataset/kg-llm/rephrased_QA_25Oct24_v2.3/train_fewshot_5_SFT.json \
    --image-folder ./dataset/kg-llm/COIN/videos/ \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode llama_3
    #--conv-mode llama_3 > ICL_KG_test_imgx4_5_shot_8B-Long1024.txt
    #--conv-mode vicuna_v1 > ICL_KG_test_imgx4_5_shot_8B-Long1024.txt


