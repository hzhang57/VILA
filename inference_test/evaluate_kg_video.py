import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, is_gemma_tokenizer, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

import requests
from io import BytesIO
import re

from llava.model.builder import load_pretrained_model
from llava.data.decord_func import decord_video_given_start_end_seconds
from llava.data.dataset import LazySupervisedDataset
from pathlib import Path

import re
from llava.eval.mmmu_utils.eval_utils import parse_choice


def parse_first_number(text):
    """
    This function takes a string as input and returns the first number it finds in the string.
    If no number is found, it returns None.
    
    :param text: The input string to search for a number
    :return: The first number found as a string, or None if no number is found
    """
    # Use regular expression to find the first number in the string
    match = re.search(r'\d+', text)
    
    # Return the matched number if found, otherwise return None
    return match.group() if match else None



class TypeAccuracy(object):
    def __init__(self, type_name):
        self.correct = 0
        self.total = 10e-9
        self.type_name = type_name

    def update(self, gt, pred):
        self.total += 1
        if "{}".format(pred) in gt:
            self.correct += 1

    def get_accuracy(self):
        return 1.0*self.correct / self.total

    def print_accuracy(self):
        #print(f"{self.type_name} Accuracy: {self.get_accuracy()} | {self.correct}/{self.total}")
        print("{} Accuracy: {:.4f} | {}/{}".format(
                self.type_name,
                self.get_accuracy(),
                self.correct,
                self.total
            ))

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def main(args):
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    #print("1 {}", model_path)
    model_name = get_model_name_from_path(model_path)
    #print("2 {}", model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
        model_name, model_base=args.model_base
        )

    # 
    #model.config.tokenizer_model_max_length = args.tokenizer_model_max_length


    ## Save outputs to list
    model_info = Path(args.model_path).name
    data_info  = Path(args.question_file).name.replace(".json", "")
    output_file = model_info + "_" + data_info + "_Results.json"
    output_dict = {}

    #features of 16xNxDim
    #features = torch.ones(128, 3000, 4096).cuda()

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

   # if 'image' in annotations[0] or 'frames' in annotations[0]:
   #     image_flag = True
   # else:
   #     image_flag = False

   # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    # Overall Accuracy for All Questions
    correct = 0
    total = 0

    global_acc = TypeAccuracy("Global")
    qa1_acc = TypeAccuracy("qa1_")
    qa2_acc = TypeAccuracy("qa2_")
    qa3_acc = TypeAccuracy("qa3_")
    #qa4_acc = TypeAccuracy("qa4_")
    #qa5_acc = TypeAccuracy("qa5_")
    qa6_acc = TypeAccuracy("qa6_")
    qa7_acc = TypeAccuracy("qa7_")
    qa8_acc = TypeAccuracy("qa8_")
    qa9_acc = TypeAccuracy("qa9_")
    qa10_acc = TypeAccuracy("qa10_")
    qa11_acc = TypeAccuracy("qa11_")
    qa12_acc = TypeAccuracy("qa12_")
    qa13_acc = TypeAccuracy("qa13_")
    qa14_acc = TypeAccuracy("qa14_")
    qa15_acc = TypeAccuracy("qa15_")
    qa16_acc = TypeAccuracy("qa16_")
    qa17_acc = TypeAccuracy("qa17_")
    qa18_acc = TypeAccuracy("qa18_")
    qa19_acc = TypeAccuracy("qa19_")
    #qa20_acc = TypeAccuracy("qa20_")

    ii = 0
    for line in tqdm(annotations, total=len(annotations)):
        #if ii > 100:
        #    break
        #ii+=1
        # Q-A Pair
        idx = line["id"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers   = conversations[1]["value"]
        index2ans = line["index2ans"]
        #all_choices = line["all_choices"]
        task_label = line["task_label"]
        step_label = line["step_label"]
        all_choices = [x for x in index2ans.keys()]
        with torch.inference_mode():
            if args.num_video_frames > 0:
                # Load Image
                video_path = os.path.join(args.image_folder, line["video"])

                
                if "start_secs" in line:
                    start_secs = line['start_secs']
                    end_secs = line['end_secs']
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=args.num_video_frames)
                else:
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path,
                        num_video_frames=args.num_video_frames)
                print(frames.shape)
                images =[  Image.fromarray(x).convert('RGB') for x in frames ]

                n_images = len(images)
                print(images[0].size, n_images)
                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)

                # replace <video> with <image>\n<image>\n....
                img_placehoder = '<image>\n' * n_images 
                #print("n_image {}".format(n_images))
                qs = qs.replace("<video>\n", img_placehoder)
                #qs = qs.replace("<video>\n", img_placehoder)
                #print("ERROR: {} {} from {}-{}s not successfully decoded".format(video_path, id, start_secs, end_secs))
                    

            #print("HEHREH ", qs)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            if args.num_video_frames > 0:
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

        # Decode output
        input_token_len = input_ids.shape[1]
        print("DEBUG input_token_len: ", input_token_len)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]



        outputs = outputs.strip()
        total += 1
        #answer_id = parse_answer(outputs)
        #answer_id = parse_first_number(outputs)
        #all_choices = ["(0)", "(1)", "(2", "(3)", "(4)"]
        answer_id = parse_choice(outputs, all_choices, index2ans)
        global_acc.update(gt_answers, answer_id)
        print("{}: {}".format(idx, qs))

        output_dict[idx] = {
                "qid": idx,
                "quest_type": quest_type,
                "qs":qs,
                "gt": gt_answers.replace(")", "").replace("(", ""),
                "task_label": task_label,
                "step_label": step_label,
                "response": outputs,
                "parser": "{}".format(answer_id),
                "index2ans": index2ans
        }
        #print("Global Accu{:.4f}.\nGT: {}\nAI: {}".format(correct*1.0/total, gt_answers, outputs))
        print("GT: {}\nAI: {}".format(gt_answers, outputs))
        #global_acc.update(gt_answers, answer_id)
        if "qa1_" in quest_type:
            qa1_acc.update(gt_answers, answer_id)
        elif "qa2_" in quest_type:
            qa2_acc.update(gt_answers, answer_id)
        elif "qa3_" in quest_type:
            qa3_acc.update(gt_answers, answer_id)
        #elif "qa4_" in quest_type:
        #    qa4_acc.update(gt_answers, answer_id)
        #elif "qa5_" in quest_type:
        #    qa5_acc.update(gt_answers, answer_id)
        elif "qa6_" in quest_type:
            qa6_acc.update(gt_answers, answer_id)
        elif "qa7_" in quest_type:
            qa7_acc.update(gt_answers, answer_id)
        elif "qa8_" in quest_type:
            qa8_acc.update(gt_answers, answer_id)
        elif "qa9_" in quest_type:
            qa9_acc.update(gt_answers, answer_id)
        elif "qa10_" in quest_type:
            qa10_acc.update(gt_answers, answer_id)
        elif "qa11_" in quest_type:
            qa11_acc.update(gt_answers, answer_id)
        elif "qa12_" in quest_type:
            qa12_acc.update(gt_answers, answer_id)
        elif "qa13_" in quest_type:
            qa13_acc.update(gt_answers, answer_id)
        elif "qa14_" in quest_type:
            qa14_acc.update(gt_answers, answer_id)
        elif "qa15_" in quest_type:
            qa15_acc.update(gt_answers, answer_id)
        elif "qa16_" in quest_type:
            qa16_acc.update(gt_answers, answer_id)
        elif "qa17_" in quest_type:
            qa17_acc.update(gt_answers, answer_id)
        elif "qa18_" in quest_type:
            qa18_acc.update(gt_answers, answer_id)
        elif "qa19_" in quest_type:
            qa19_acc.update(gt_answers, answer_id)
        else:
            print(f"Unknown Type: {idx}")
        # print each type accuracy
        print("-----"*5)
        global_acc.print_accuracy()
        print("-----"*5)
        qa1_acc.print_accuracy()
        qa2_acc.print_accuracy()
        qa3_acc.print_accuracy()
        #qa4_acc.print_accuracy()
        #qa5_acc.print_accuracy()
        qa6_acc.print_accuracy()
        qa7_acc.print_accuracy()
        qa8_acc.print_accuracy()
        qa9_acc.print_accuracy()
        qa10_acc.print_accuracy()
        qa11_acc.print_accuracy()
        qa12_acc.print_accuracy()
        qa13_acc.print_accuracy()
        qa14_acc.print_accuracy()
        qa15_acc.print_accuracy()
        qa16_acc.print_accuracy()
        qa17_acc.print_accuracy()
        qa18_acc.print_accuracy()
        qa19_acc.print_accuracy()
        print("-----"*5)
        # average over type
        avg_acc = (qa1_acc.get_accuracy() + qa2_acc.get_accuracy() + qa3_acc.get_accuracy() + qa6_acc.get_accuracy() + qa7_acc.get_accuracy() + qa8_acc.get_accuracy() + qa9_acc.get_accuracy() + qa10_acc.get_accuracy() + qa11_acc.get_accuracy() + qa12_acc.get_accuracy() + qa13_acc.get_accuracy() + qa14_acc.get_accuracy() + qa15_acc.get_accuracy() +qa16_acc.get_accuracy() + qa17_acc.get_accuracy() + qa18_acc.get_accuracy() + qa19_acc.get_accuracy()) / 17.0
        print("Average Acc over Type: {:.4f}".format(avg_acc))


   # eval_dict ={
   #     "interact_acc":interact_acc.get_accuracy(),
   #     "sequence_acc": sequence_acc.get_accuracy(),
   #     "predict_acc":predict_acc.get_accuracy(),
   #     "feasibility_acc": feasibility_acc.get_accuracy(),
   #     "avg_acc":avg_acc,
   #         }

   # result_file = "./result_star_video/" + Path(args.model_path).name + "Fx{}.json".format(args.num_video_frames)


   # print("save to {}".format(result_file))
   # with open(result_file, "w") as f:
   #     json.dump(eval_dict, f, indent=2)

   # final_out = {
   #     "Interaction": interact_results,
   #     "Sequence": sequence_results,
   #     "Prediction": predict_results,
   #     "Feasibility": feasibility_results
   #         }

    print("save to {}".format(output_file))
    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=2)


    print("Process Finished")

def parse_answer(outputs):
    if "Answer is:" in outputs:
    # with graph
        outputs = outputs.split("Answer is: ")[-1]
    if "answer is " in outputs:
    # with graph
        outputs = outputs.split("answer is ")[-1].strip(".")
    #print("XXXXXXXXXXX", outputs)
    # remove graph
    answer_id = outputs[0]
    try:
        answer_id = int(answer_id)
        return answer_id
    except:
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_video_frames", type=int, default=1)
    #parser.add_argument("--tokenizer_model_max_length", type=int, default=8192)
    args = parser.parse_args()
    main(args)
