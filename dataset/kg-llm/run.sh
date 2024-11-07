#python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2/testing.json
#python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2/validation.json
#python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2/training.json
python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_1.json
python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_5.json
python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_10.json
python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_50.json
python gen_sft_json.py --input rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_100.json
