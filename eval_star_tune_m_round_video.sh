#!/bin/bash
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Video_Gen_Program_vila3b_loss2
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Video_Gen_Program_Middle_vila3b_loss2
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Video_Gen_Program_Middle_OptEarly_vila3b_loss2
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Video_Gen_Program_Middle_NoTab_vila3b_loss2
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Video_Gen_Program_Middle_NoTabOptEarly_vila3b_loss2
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Video_Gen_Program_Middle_NoTabOptEarlyFix_vila3b_loss2
CKPT=result_kg

python inference_test/evaluate_star_m_round_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 4 \
    --question-file ./dataset/star/sft_annots_video/STAR_val_NEAT_Query_Video_Gen_Program_Middle_NoTabOptEarlyFix_v5.4.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1

