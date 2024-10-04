#!/bin/bash
MODEL_PATH=Efficient-Large-Model/VILA1.5-3B
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Program_Video_vila3b_loss2
MODEL_PATH=Efficient-Large-Model/Llama-3-VILA1.5-8B
MODEL_PATH=Efficient-Large-Model/VILA1.5-13b
CKPT=result_kg

python inference_test/evaluate_star_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 4 \
    --question-file ./dataset/star/sft_annots_video/STAR_val_NEAT_Query_Video_v3.0.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1

