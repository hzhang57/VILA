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

import difflib

def print_gt_and_answers(a, b):
    # Split the strings into lines
    lines_a = a.split('\n')
    lines_b = b.split('\n')
    
    # Get the maximum width for each column to align the output
    max_len_a = max(len(line) for line in lines_a)
    max_len_b = max(len(line) for line in lines_b)
    
    # Print header
    print(f"{'----gt-program----'.ljust(max_len_a)}|{'----gen-program----'.ljust(max_len_b)}")
    
    # Iterate through both lists of lines and print them side by side
    for line_a, line_b in zip(lines_a, lines_b):
        # Print each line from a and b aligned
        print(f"{line_a.ljust(max_len_a)}|{line_b.ljust(max_len_b)}")
    
    # Handle any remaining lines if a and b are of different lengths
    if len(lines_a) > len(lines_b):
        for line_a in lines_a[len(lines_b):]:
            print(f"{line_a.ljust(max_len_a)}|{' ' * max_len_b}")
    elif len(lines_b) > len(lines_a):
        for line_b in lines_b[len(lines_a):]:
            print(f"{' ' * max_len_a}|{line_b.ljust(max_len_b)}")
    print('--'*40)

def string_diff(string1, string2):
    # Use difflib.ndiff to find differences
    diff = difflib.ndiff(string1, string2)
    return '\n'.join(diff)


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

    def update_program(self, gt, pred):
        self.total += 1
        if "{}".format(pred) == gt:
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


def conv_pred(prompt, conv, tokenizer, model, use_image, IMAGE_TOKEN_INDEX, images_tensor=None):
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
    #print("DEBUG input_token_len: ", input_token_len)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()

    return outputs


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
    qa0_acc = TypeAccuracy("Program Accuracy")
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

        # Gen Program
        conversations = line["conversations"]
        qs_0 = conversations[0]["value"]
        gt_answers_0   = conversations[1]["value"]

        # Gen Answer
        qs_1 = conversations[2]["value"]
        gt_answers_1   = conversations[3]["value"]

        index2ans = line["index2ans"]
        all_choices = line["all_choices"]
        
        use_image = False
        with torch.inference_mode():
            if args.num_video_frames > 0:
                use_image = True
                # Load Image
                video_path = os.path.join(args.image_folder, line["video"])
                
                if "start_secs" in line:
                    start_secs = line['start_secs']
                    end_secs = line['end_secs']
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=args.num_video_frames)
                    print("st-ed {}-{}".format(start_secs, end_secs))
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
            else:
                n_images=0
                    
            total += 1
            #Program
            img_placehoder = '<image>\n' * n_images
            qs_0 = qs_0.replace("<video>\n", img_placehoder)

            conv_0 = conv_templates[args.conv_mode].copy()
            conv_0.append_message(conv_0.roles[0], qs_0)
            conv_0.append_message(conv_0.roles[1], None)
            prompt_0 = conv_0.get_prompt()
            
            outputs_0 = conv_pred(prompt_0, conv_0, tokenizer, model, use_image, IMAGE_TOKEN_INDEX, images_tensor)    
            answer_id_0 = outputs_0
            
            # Remove last \n in gt_answers_0, 2024/10/13
            gt_answers_0 = gt_answers_0.strip("\n")
            answer_id_0  = answer_id_0.strip("\n")

            qa0_acc.update_program(gt_answers_0, answer_id_0)
            print("{}: {}".format(idx, qs_0))
            print("[WARNING]: Program Matched AI ? {}".format(gt_answers_0 == outputs_0))
            print_gt_and_answers(gt_answers_0, outputs_0)
            #print("======Program GT  {}".format(gt_answers_0))
            #print("======Program GEN {}".format(outputs_0))
            #print("Program Matched Diff {} {}".format(len(gt_answers_0), len( outputs_0)))

            # Answer
            conv_1 = conv_templates[args.conv_mode].copy()
            conv_1.append_message(conv_1.roles[0], qs_0)
            conv_1.append_message(conv_1.roles[1], answer_id_0)
            conv_1.append_message(conv_1.roles[0], qs_1)
            conv_1.append_message(conv_1.roles[1], None)
            prompt_1 = conv_1.get_prompt()

            outputs_1 = conv_pred(prompt_1, conv_1, tokenizer, model, use_image, IMAGE_TOKEN_INDEX, images_tensor)    
            answer_id_1 = parse_choice(outputs_1, all_choices, index2ans)
            global_acc.update(gt_answers_1, answer_id_1)
            #print("{}: {}\nProgram: {}\n{}".format(idx, qs_0, answer_id_0, qs_1))
            print("{}: {}{}{}".format(idx, qs_0, answer_id_0, qs_1))
            print("GT: {}\nAI: {}".format(gt_answers_1, outputs_1))
            #print("Global Accu{:.4f}.\nGT: {}\nAI: {}".format(correct*1.0/total, gt_answers, outputs))
            if "Interaction" in quest_type:
                qa1_acc.update(gt_answers_1, answer_id_1)
            elif "Sequence" in quest_type:
                qa2_acc.update(gt_answers_1, answer_id_1)
            elif "Prediction" in quest_type:
                qa3_acc.update(gt_answers_1, answer_id_1)
            elif "Feasibility" in quest_type:
                qa4_acc.update(gt_answers_1, answer_id_1)
            else:
                print(f"Unknown Type: {idx}")
            # print each type accuracy
            print("--"*40)
            qa0_acc.print_accuracy()
            print("--"*40)
            qa1_acc.print_accuracy()
            qa2_acc.print_accuracy()
            qa3_acc.print_accuracy()
            qa4_acc.print_accuracy()
            # average over type
            avg_acc = (qa1_acc.get_accuracy() + qa2_acc.get_accuracy() + qa3_acc.get_accuracy() + qa4_acc.get_accuracy() ) / 4.0
            print("Average Acc over Type: {:.4f}".format(avg_acc))
            print("--"*40)

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
