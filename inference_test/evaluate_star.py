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
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
        model_name, model_base=args.model_base
        )

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    correct = 0
    total = 0

    global_acc = TypeAccuracy("Global")
    qa1_acc = TypeAccuracy("Interact")
    qa2_acc = TypeAccuracy("Sequence")
    qa3_acc = TypeAccuracy("Predict")
    qa4_acc = TypeAccuracy("Feasibility")

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
        all_choices = line["all_choices"]
        
        use_image = False
        with torch.inference_mode():
            if args.num_video_frames > 0:
                # Load Image
                if 'image' in line:
                    image_files = [os.path.join(args.image_folder, image) for image in [line["image"]]]
                    use_image = True
                elif 'frames' in line:
                    image_files = [os.path.join(args.image_folder, image) for image in line["frames"]]
                    use_image = True
                else:
                    #not implementd
                    use_image = False
                    print("Not implemented yet")
                if use_image:
                    images = load_images(image_files)
                    n_images = len(images)
                    print(images[0].size, n_images)
                    images_tensor = process_images(
                        images,
                        image_processor,
                        model.config
                    ).to(model.device, dtype=torch.float16)
                    
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
            if use_image:
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
        answer_id = parse_choice(outputs, all_choices, index2ans)
        global_acc.update(gt_answers, answer_id)
        print("{}: {}".format(idx, qs))
        #print("Global Accu{:.4f}.\nGT: {}\nAI: {}".format(correct*1.0/total, gt_answers, outputs))
        print("GT: {}\nAI: {}".format(gt_answers, outputs))
        if "Interaction" in quest_type:
            qa1_acc.update(gt_answers, answer_id)
        elif "Sequence" in quest_type:
            qa2_acc.update(gt_answers, answer_id)
        elif "Prediction" in quest_type:
            qa3_acc.update(gt_answers, answer_id)
        elif "Feasibility" in quest_type:
            qa4_acc.update(gt_answers, answer_id)
        else:
            print(f"Unknown Type: {idx}")
        # print each type accuracy
        print("-----"*5)
        qa1_acc.print_accuracy()
        qa2_acc.print_accuracy()
        qa3_acc.print_accuracy()
        qa4_acc.print_accuracy()
        print("-----"*5)
        # average over type
        avg_acc = (qa1_acc.get_accuracy() + qa2_acc.get_accuracy() + qa3_acc.get_accuracy() + qa4_acc.get_accuracy() ) / 4.0
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    print("Process Finished")

def parse_answer(outputs):
    if "Answer is:" in outputs:
    # with graph
        outputs = outputs.split("Answer is: ")[-1]
    if "answer is " in outputs:
    # with graph
        outputs = outputs.split("answer is ")[-1].strip(".")
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
