python scripts/merge_lora_weights.py \
	--model-path ./ckpts_vid/kg_qa_imgx{4}_3_shot_vila3b_loss2_epoch100_lora_v2 \
	--model-base Efficient-Large-Model/VILA1.5-3B \
	--save-model-path ./ckpts_vid/kg_qa_imgx{4}_3_shot_vila3b_loss2_epoch100_lora_v2_Merged
