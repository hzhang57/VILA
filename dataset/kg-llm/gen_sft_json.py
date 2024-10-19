import os
import json
from pathlib import Path
import argparse
import re
from tqdm import tqdm

def file_exists(filepath):
    if os.path.exists(filepath):
        return True
    else:
        print("The file does not exist.")
        return False
def add_spaces_to_camel_case(text):
    # Example usage:
    #print(add_spaces_to_camel_case("InstallCeramicTile"))  # Output: "Install Ceramic Tile"
    #print(add_spaces_to_camel_case("Install Creaamic Tile"))  # Output: "Install Creaamic Tile"
    # Check if the input already contains spaces
    if " " in text:
        return text
    # Add a space before each uppercase letter (except the first one) and join them
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)



def find_video_path_by_id(video_id, video_folder):
    # List all subfolders in video_folder
    video_folder = Path(video_folder)
    subfolders = [f.name for f in video_folder.iterdir() if f.is_dir()]

    # Iteratre subfolders, and find the video under subfolder
    for subfolder in subfolders:
        # Iterate all files in subfolder
        for file in (video_folder / subfolder).rglob('*'):
            #print(file, video_id)
            if video_id in file.name:
                return str(subfolder / Path(file.name))

    return "None"

def get_file_name(one_type, folder):
    file_name = ""
    pattern_to_match = "qa{}_".format(one_type)
    
    # list all json file under folder
    folder = Path(folder)
    json_files = list(folder.glob("*.json"))
    
    for json in json_files:
        json = str(json)
        if pattern_to_match in json:
            file_name = json     
    return file_name


def load_json(one_file, miss_vid_file):
    with open(miss_vid_file, "r") as f:
        lines = f.readlines()
        miss_list = [ a.strip() for a in lines]

    annots = ""
    with open(one_file, "r") as f:
        annots = json.load(f)
    
    sft_annots = []
    #for ii, one_line in enumerate(annots):
    for one_line in tqdm(annots, desc="Processing annotations"):
        video_id = one_line['video_id']
        if video_id in miss_list:
            print("Find miss {}".format(video_id))
            continue # video missing in annotations
        else:
            one_sample = {}

            question = one_line['question']
            options  = one_line['options']
            answer   = one_line['answer']
            step_id  = one_line['step']['id']
            start_secs = one_line['step']['segment'][0]
            end_secs   = one_line['step']['segment'][1]
            question_type = one_line['quest_type']

            one_sample["id"] = "{}_{}".format(video_id, step_id)
            one_sample["video"] = find_video_path_by_id(video_id, "./COIN/videos/")

            index2ans = {}
            opts = ""
            for ii, one_opt in enumerate(options):
                one_opt = add_spaces_to_camel_case(one_opt)
                opts += ("({}) {}; ".format(ii, one_opt))
                index2ans[str(ii)] = one_opt
            opts = opts.rstrip("; ")
            question = "<video>\n{} select from options: {}.".format(question, opts)

            one_sample["conversations"] = [
                {
                "from": "human",
                "value": question
                },
                {
                "from":"gpt",
                "value": "({}) {}".format(answer, add_spaces_to_camel_case(options[answer]))
                }
            ]
            one_sample["quest_type"] = question_type
            one_sample["start_secs"] = start_secs
            one_sample["end_secs"]   = end_secs
            one_sample["index2ans"]   = index2ans
            sft_annots.append(one_sample)

    return sft_annots


def main():
    # add argument
    parser = argparse.ArgumentParser("Convert KG-LLM to ViLA format")
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    if not file_exists(args.input):
        exit()
    else:
        output_json = args.input.replace(".json", "_SFT.json")
    miss_vid_file = "./miss_vid_list_1494.txt"

    print("Processing {}...".format(args.input))
    
    out_content = load_json(args.input, miss_vid_file)    

    with open(output_json, "w") as f:
        json.dump(out_content, f, indent=2)
    



    print("Process Finished")

if __name__ == "__main__":
    main()
