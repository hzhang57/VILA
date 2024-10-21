import os
import json
import random
from pathlib import Path

import numpy as np

random.seed(42)  # Set the seed to some fixed value
np.random.seed(42)  # Set the seed to some fixed value

MODULES = {
            'Situations':{'nargs': 0},
            'Actions': {'nargs': 1},
            'Objs':{'nargs': 1},
            'Rels':{'nargs': 1},

            'Filter_Actions_with_Verb': {'nargs': 2},
            'Filter_Actions_with_Obj': {'nargs': 2},
            'Filter_After_Actions':{'nargs': 2},
            'Filter_Before_Actions':{ 'nargs': 2},
            'Filter_Situations_with_Rel':{'nargs': 2},
            'Filter_Situations_with_Obj':{'nargs': 2},
            'Filter_Objs_by_Verb':{'nargs': 2},
            
            'Unique': {'nargs': 1},
            'Equal': {'nargs': 2},
            'Union':{'nargs': 2},
            'Belong_to':{'nargs': 2},
            'Except':{'nargs': 2},

            'Query_Objs': {'nargs': 1},
            'Query_Actions': {'nargs': 1},
            'Query_Verbs': {'nargs': 1},
            'Query_Earliest_Action':{'nargs': 1},
            'Query_Latest_Action':{'nargs': 1},
        }



def pg_transform(raw_pg, program_middle):
    pg = []
    for ii, p in enumerate(raw_pg):
        m_program  = program_middle[ii]['function'] 
        m_return   = program_middle[ii]['return_value'] 
        print("m_program == gt_program {}".format(m_program==p['function']))
        assert m_program==p['function'], exit()
        result_pad = ":={}".format(list(set(m_return)))  
        pg.extend(p['value_input'])
        pg.extend(["{}{}".format(p['function'], result_pad)])
    return pg

#def format_code_string(code_str):
#    indent_level = 0
#    formatted_code = ""
#    i = 0
#
#    while i < len(code_str):
#        char = code_str[i]
#
#        if char == '(':
#            if i + 1 < len(code_str) and code_str[i + 1] == ')':
#                # Handle empty parentheses
#                formatted_code += " ()"
#                i += 1
#            else:
#                indent_level += 1
#                formatted_code += " (\n" + "    " * indent_level
#        elif char == ')':
#            formatted_code += "\n" + "    " * indent_level + ")"
#            indent_level -= 1
#        elif char == ',':
#            formatted_code += ",\n" + "    " * indent_level
#        elif char == ':':
#            # Handle :=
#            if i + 1 < len(code_str) and code_str[i + 1] == '=':
#                formatted_code += " :="
#                i += 1
#            else:
#                formatted_code += ':'
#        else:
#            formatted_code += char
#
#        i += 1
#
#    return formatted_code


def format_code_string(code_str):
    indent_level = 0
    formatted_code = ""
    i = 0

    while i < len(code_str):
        char = code_str[i]

        if char == '(':
            if i + 1 < len(code_str) and code_str[i + 1] == ')':
                # Handle empty parentheses
                formatted_code += " ()"
                i += 1
            else:
                indent_level += 1
                formatted_code += " (\n" + "\t" * indent_level
        elif char == ')':
            formatted_code += "\n" + "\t" * indent_level + ")"
            indent_level -= 1
        elif char == ',':
            formatted_code += ",\n" + "\t" * indent_level
        elif char == ':':
            # Handle :=
            if i + 1 < len(code_str) and code_str[i + 1] == '=':
                formatted_code += " :="
                i += 1
            else:
                formatted_code += ':'
        else:
            formatted_code += char

        i += 1

    return formatted_code



def convert_program_v7(program, program_middle):
    exe_stack = []
    old_print = ""
    #print("Program ", program)
    tl = len(program)
    print("True Program ", program)
    print("Midd {}".format(program_middle))
    pg = pg_transform(program, program_middle)

    for ii, m in enumerate(pg):
        #print("XXXXX", m)
        if ":=" in m:
            subs = m.split(":=") # parse function and its return
            #print("XXXXXXXXX", subs)
            m = subs[0].strip()
            m_return = subs[1].strip()
            if m_return != "[]":
                m_return = ":= {}".format(m_return)
            else:
                m_return = ""
        else:
            m_return = ""

        if m not in MODULES:
            exe_stack.append(m)
            #print(exe_stack)
        else:
            argv = []
            nargs = MODULES[m]['nargs']
            #print(nargs)
            for jj in range(nargs):
                #print("HERE")
                #print(exe_stack)
                if exe_stack:
                    argv.insert(0, exe_stack.pop())
                    argv_lines = ",".join(argv)
                    #print("argv ", ",".join(argv))
            
            if argv:
                old_print = "{}({})".format(m, argv_lines)
            else:
                old_print = "{}()".format(m)
            

            exe_stack.append(old_print)

    #print(exe_stack[-1])
    program_structured = format_code_string(exe_stack[-1])
    program_structured = remove_whitespace_in_brackets(program_structured)

    return program_structured


def convert_subs_program_v7(program, program_middle):
    subs_program = []
    exe_stack = []
    old_print = ""
    #print("Program ", program)
    tl = len(program)
    print("True Program ", program)
    print("Midd {}".format(program_middle))
    pg = pg_transform(program, program_middle)

    for ii, m in enumerate(pg):
        #print("XXXXX", m)
        if ":=" in m:
            subs = m.split(":=") # parse function and its return
            #print("XXXXXXXXX", subs)
            m = subs[0].strip()
            m_return = subs[1].strip()
            if m_return != "[]":
                m_return = ":= {}".format(m_return)
            else:
                m_return = ""
        else:
            m_return = ""

        if m not in MODULES:
            exe_stack.append(m)
            #print(exe_stack)
        else:
            argv = []
            nargs = MODULES[m]['nargs']
            #print(nargs)
            for jj in range(nargs):
                #print("HERE")
                #print(exe_stack)
                if exe_stack:
                    argv.insert(0, exe_stack.pop())
                    argv_lines = ",".join(argv)
                    #print("argv ", ",".join(argv))
            
            if argv:
                old_print = "{}({}){}".format(m, argv_lines, m_return)
            else:
                old_print = "{}(){}".format(m, m_return)
        
            if m_return != "":
                subs_program.append(old_print)


            exe_stack.append(old_print)

    #print(exe_stack[-1])

    #return exe_stack[-1]
    subs_program = [ format_code_string(x) for x in subs_program]
    subs_program = [ remove_whitespace_in_brackets(x) for x in subs_program]
    return subs_program

def remove_after_bracket(input_string):
    # Find the position of the closing bracket "]"
    end_index = input_string.find(']')
    # If "]" is found, return the substring up to and including the bracket
    if end_index != -1:
        return input_string[:end_index + 1]
    else:
        # If no "]" is found, return the original string
        return input_string


import re

def remove_whitespace_in_brackets(input_str):
    def replacer(match):
        content = match.group(1)
        cleaned_content = re.sub(r'[\n\t]', '', content)
        return f'[{cleaned_content}]'

    result = re.sub(r'\[(.*?)\]', replacer, input_str, flags=re.DOTALL)
    result = merge_consecutive_spaces(result)
    return result
    
def merge_consecutive_spaces(input_string):
    # This regex will match any content between [ and ] and apply another regex to replace multiple spaces with a single space
    def merge_spaces(match):
        content = match.group(0)
        # Replace multiple spaces with a single space inside [ ]
        return re.sub(r'\s{2,}', ' ', content)
    
    # Apply the regex to find all instances between [ and ]
    result = re.sub(r'\[.*?\]', merge_spaces, input_string, flags=re.DOTALL)
    result = result.replace(" (", "(")
    return result


    
    
def format_into_instruct(sample, image_used, qid_to_frames, 
    obj_dict, rel_dict, act_dict, middle_dict):
    out = {}
    out['id'] = sample['question_id']
    program_middle = middle_dict[out['id']]

    quest_type = get_quest_type(out['id'])
    question = sample['question']
    if "answer" in sample.keys():
        answer = sample['answer']
    else:
        answer = "0 null"
        answer_id = 0
    
    program = sample['question_program']
    program_structured = convert_program_v7(program, program_middle)
    #program_structured = format_program_string(program_structured)

    subs_program = convert_subs_program_v7(program, program_middle)


    print("****"*8)
    print(program_structured)

    video_id = sample['video_id']
    start_sec = sample['start']
    end_sec   = sample['end']
    out['video'] = "{}.mp4".format(video_id)
    out['start_secs'] = start_sec
    out['end_secs'] = end_sec

    # 2. Create Images List
    image_list = qid_to_frames[out['id']]
    if len(image_list) > image_used:
        # random select image_used amount
        rand_index = random.sample(range(0, len(image_list)-1), image_used)
        sorted_index  = sorted(rand_index)
        sorted_images = sorted(image_list)
        frames = []
        for jj in sorted_index:
            frames.append(sorted_images[jj])
        image_list = frames

    #if image_used == 1:
    #    #out['image'] = image_list[0]
    #    num_images = 1
    #else:
    #    #out['frames'] = image_list
    #    num_images = len(out['frames'])

    # 1. Create Conversations
    # Get all choices:
    choices = sample['choices']
    options = []
    all_choices = []
    index2ans = {}
    for choice in choices:
        options.append("({}) {}\n".format(choice['choice_id'], choice['choice'].strip(".")))
        #options.append("{};".format(choice['choice'].strip(".")))
        all_choices.append(str(choice['choice_id']))
        index2ans[str(choice['choice_id'])] = choice['choice'].strip(".").replace("/", ', ')#.lower()
        if choice['choice'] == answer:
            answer_id = choice['choice_id']
    options = ''.join(options).replace('/', ', ')

    #from_human = "<image>\n" * num_images + "STAR Task: " + program + " Select from options: " + options
    #from_human = "<video>\n" + "STAR Task: {} ".format(question)  + program_structured + " Select from options: " + options
    #from_human_0 = "<video>\n" + "STAR Task:\n{}\nSelect from options: {}".format(program_structured, options)
    from_human_0 = "<video>\n" + "{}\nProgram:\n".format(question)
    from_gpt_0 = program_structured
    from_human_1 = "\nSelect from options:\n" + options
    #from_human_1 = from_human_1.strip(";") + "."
    from_gpt_1   = "{} {}".format(answer_id, answer.replace('/', ', '))
    conversation = [
        {
        "from": "human",
        "value": from_human_0
        },
        {
        "from": "gpt",
        "value": from_gpt_0
        },
        {
        "from": "human",
        "value": from_human_1
        },
        {
        "from": "gpt",
        "value": from_gpt_1
        }
    ]

    ## Add Subs Program
    for x in subs_program:
        print(x)
        subs = x.split(" := ")
        from_human_subs = subs[0].strip() + "\n"
        from_gpt_subs   = subs[1].strip() 
        from_gpt_subs = remove_after_bracket(from_gpt_subs) + "\n"
        conversation.append(
            {
                "from": "human",
                "value": from_human_subs
            })
        conversation.append(
            {
                "from": "gpt",
                "value": from_gpt_subs
            }
        )

    out['conversations'] = conversation
    out['all_choices'] = all_choices

    out["index2ans"] = index2ans
    out["quest_type"] = quest_type

    return out

def generate_random_numbers(n, range_start, range_end):
    return [random.randint(range_start, range_end) for _ in range(n)]
    
def gen_classes_dict(action_classes_file):
    action_classes_dict = {}
    with open(action_classes_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        action_classes_dict[line[0]] = line[1].replace("/", " / ")
    print(action_classes_dict)
    return action_classes_dict


def gen_action_classes_map_dict(action_mapping_file, obj_dict, verb_dict):
    action_classes_map_dict = {}
    with open(action_mapping_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        verb, obj = line[1].strip(), line[2].strip()
        action_classes_map_dict[line[0]] = "{} {}".format(verb_dict[verb], obj_dict[obj])

    #print(action_classes_map_dict)
    return action_classes_map_dict


def get_quest_type(a_string):
    subs = a_string.split("_")
    quest_type = subs[0].strip()
    return quest_type


def get_output_name(base_folder, input_json, image_used, random_order=False, reverse_order=False, ver=3.0):
    """
    Generate an output file name based on the input parameters.
    
    Args:
    base_folder (str): The directory where the output file will be saved.
    input_json (str): The path to the input JSON file.
    image_used (str): Identifier for the image used which will be part of the file name.
    random_order (bool): If True, append '_random' to the file name before the extension.
    reverse_order (bool): If True, append '_reverse' to the file name before the extension.
    
    Returns:
    str: The formatted output file name.
    """
    # Extract the name from the input JSON path
    input_name = Path(input_json).stem

    # Format the initial output file name
    output_name = f"{base_folder}/{input_name}_{ver}.json"
    
    # Append modifiers based on the order flags
    if random_order:
        output_name = output_name.replace(".json", "_random.json")
    if reverse_order:
        output_name = output_name.replace(".json", "_reverse.json")
    
    return output_name
def main():
    class_dict = {}
    action_classes_file = "star_annotations/classes/action_classes.txt"
    action_classes_dict = gen_classes_dict(action_classes_file)
    verb_classes_file   = "star_annotations/classes/verb_classes.txt"
    verb_classes_dict = gen_classes_dict(verb_classes_file)
    obj_classes_file    = "star_annotations/classes/object_classes.txt"
    obj_classes_dict = gen_classes_dict(obj_classes_file)
    relation_classes_file    = "star_annotations/classes/relationship_classes.txt"
    relation_classes_dict = gen_classes_dict(relation_classes_file)
    action_mapping_file = "star_annotations/classes/action_mapping.txt"
    action_classes_map_dict = gen_action_classes_map_dict(action_mapping_file, obj_classes_dict, verb_classes_dict)
    act_dict = action_classes_dict
    verb_dict = verb_classes_dict
    obj_dict = obj_classes_dict
    rel_dict = relation_classes_dict
    
    #csv frames list
    import csv
    csv_file = "star_annotations/Video_Keyframe_IDs.csv"
    qid_to_frames = {}
    with open(csv_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            qid = row[0].strip()
            vid = row[1].strip()

            pth_list = []
            frames = row[2].replace("[","").replace("]","")
            frames = frames.split(",")
            for frame in frames:
                frame = frame.strip().replace("'","" )
                frame_pth = "{}.mp4/{}.png".format(vid, frame)
                pth_list.append(frame_pth)
            qid_to_frames[qid] = pth_list
            #print(qid_to_frames[qid])
    #exit()

    middle_val_file = "./program_executor/val_out_v2_program_results.json"
    middle_train_file = "./program_executor/train_out_v2_program_results.json"

    with open(middle_val_file, "r") as f:
        middle_dict = json.load(f)

    with open(middle_train_file, "r") as f:
        middle_dict.update(json.load(f))

    image_used = 4
    random_order = False
    reverse_order= False
    ver = 3.0
    base_folder = "../sft_annots_video_v3.1"
    input_json_s = [ "star_annotations/STAR_val_NEAT.json", "star_annotations/STAR_train_NEAT.json"]

    for input_json in input_json_s:
        output =  get_output_name(base_folder, input_json, image_used, random_order, reverse_order, "Query_Gen_Program_Middle_Video_3Stage_v3.1")

        print("Save new annotations to {}".format(output))
        out_list = []
        with open(input_json, 'r') as f:
            data = json.load(f)
        print("Data Loaded")

        for i in range(len(data)):
            out = {}
            a_sample = data[i]
            a_sample = format_into_instruct(a_sample, image_used, qid_to_frames, obj_dict, rel_dict, act_dict, middle_dict)


            # Reverse frame order

            out_list.append(a_sample)

        # Save the output in beautiful json format
        with open(output, 'w') as f:
            json.dump(out_list, f, indent=2, ensure_ascii=False)

    print("Process Finished")

if __name__ == "__main__":
    main()
