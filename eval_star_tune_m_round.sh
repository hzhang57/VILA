#!/bin/bash
MODEL_PATH=ckpts/star_qa_imgx4_Query_Image_Gen_Program_vila3b_loss2
CKPT=result_kg

python inference_test/evaluate_star_m_round.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 4 \
    --question-file ./dataset/star/sft_annots/STAR_val_NEAT_imgx4_Query_Image_Gen_Program_v3.0.json \
    --image-folder ./dataset/star/charadesv1_480/frames \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1

