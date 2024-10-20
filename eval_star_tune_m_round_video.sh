#!/bin/bash
MODEL_PATH=ckpts_vid/star_qa_imgx{4}_Query_Gen_Program_Video_vila3b_loss2_e1_3.1
CKPT=result_kg

python inference_test/evaluate_star_m_round_video.py \
    --model-path $MODEL_PATH \
    --max_new_tokens 512 \
    --num_video_frames 4 \
    --question-file ./dataset/star/sft_annots_video_v3.1/STAR_val_NEAT_Query_Gen_Program_Video_v3.1.json \
    --image-folder ./dataset/star/charadesv1_480/video \
    --answers-file ./tmp/$MODEL_PATH \
    --temperature 0 \
    --conv-mode vicuna_v1 > TUNE_STAR_val_NEAT_Query_Gen_Program_Video_v3.1.txt

