# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    sharegpt4v_pretrain = Dataset(
        dataset_name="sharegpt4v_pretrain",
        dataset_type="torch",
        data_path="",
        image_path="",
        description="Original data source: https://sharegpt4v.github.io/ ~1M long Image - Text pair generated by ShareGPT4V captioner.",
    )
    add_dataset(sharegpt4v_pretrain)
    
    sharegpt4v_sft = Dataset(
        dataset_name="sharegpt4v_sft",
        dataset_type="torch",
        data_path="",
        image_path="",
        description="Original data source: https://sharegpt4v.github.io/ 655K llava_1_5_sft data relablled w/ ShareGPT4V captioner.",
    )
    add_dataset(sharegpt4v_sft)

    # Add STAR-QA dataset by Hao Zhang
    # Program + Image -> Answer
    #star_qa_imgx4_Program_Image = Dataset(
    #    dataset_name="star_qa_imgx4_Program_Image",
    #    dataset_type="star_qa",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Program_Image_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Program_Image)

    ## Query + Image -> Program
    ## Query + Program + Image -> Answer
    #star_qa_imgx4_Query_Image_Gen_Program = Dataset(
    #    dataset_name="star_qa_imgx4_Query_Image_Gen_Program",
    #    dataset_type="star_qa",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Query_Image_Gen_Program_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Query_Image_Gen_Program)

    ## Query + Image -> Answer
    #star_qa_imgx4_Query_Image = Dataset(
    #    dataset_name="star_qa_imgx4_Query_Image",
    #    dataset_type="star_qa",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Query_Image_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Query_Image)

    ## Query + Program + Graph + Image -> Answer
    #star_qa_imgx4_Query_Program_Graph_Image = Dataset(
    #    dataset_name="star_qa_imgx4_Query_Program_Graph_Image",
    #    dataset_type="star_qa",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Query_Program_Graph_Image_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Query_Program_Graph_Image)

    ## Query + Program + Image -> Answer
    #star_qa_imgx4_Query_Program_Image = Dataset(
    #    dataset_name="star_qa_imgx4_Query_Program_Image",
    #    dataset_type="star_qa",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Query_Program_Image_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Query_Program_Image)


    ## Program + Graph -> Answer
    #star_qa_imgx4_Program_Graph = Dataset(
    #    dataset_name="star_qa_imgx4_Program_Graph",
    #    dataset_type="torch",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Program_Graph_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Program_Graph)


    ## Query + Graph -> Answer
    #star_qa_imgx4_Query_Graph = Dataset(
    #    dataset_name="star_qa_imgx4_Query_Graph",
    #    dataset_type="torch",
    #    data_path="./dataset/star/sft_annots/STAR_train_NEAT_imgx4_Query_Graph_v3.0.json",
    #    image_path="./dataset/star/charadesv1_480/frames/",
    #    description="Original data source: https://bobbywu.com/STAR/",
    #)
    #add_dataset(star_qa_imgx4_Query_Graph)

    ## Directly use video
    # Query + Video -> Answer
    kg_qa_Query_Video_1_shot = Dataset(
        dataset_name="kg_qa_Query_Video_1_shot",
        dataset_type="star_qa_decord",
        data_path="./dataset/kg-llm/rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_1_SFT.json",
        image_path="./dataset/kg-llm/COIN/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(kg_qa_Query_Video_1_shot)

    kg_qa_Query_Video_5_shot = Dataset(
        dataset_name="kg_qa_Query_Video_5_shot",
        dataset_type="star_qa_decord",
        data_path="./dataset/kg-llm/rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_5_SFT.json",
        image_path="./dataset/kg-llm/COIN/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(kg_qa_Query_Video_5_shot)

    kg_qa_Query_Video_10_shot = Dataset(
        dataset_name="kg_qa_Query_Video_10_shot",
        dataset_type="star_qa_decord",
        data_path="./dataset/kg-llm/rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_10_SFT.json",
        image_path="./dataset/kg-llm/COIN/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(kg_qa_Query_Video_10_shot)

    kg_qa_Query_Video_50_shot = Dataset(
        dataset_name="kg_qa_Query_Video_50_shot",
        dataset_type="star_qa_decord",
        data_path="./dataset/kg-llm/rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_50_SFT.json",
        image_path="./dataset/kg-llm/COIN/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(kg_qa_Query_Video_50_shot)

    kg_qa_Query_Video_100_shot = Dataset(
        dataset_name="kg_qa_Query_Video_100_shot",
        dataset_type="star_qa_decord",
        data_path="./dataset/kg-llm/rephrased_QA_25Oct24_v2.2_FewShot/training_fewshot_100_SFT.json",
        image_path="./dataset/kg-llm/COIN/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(kg_qa_Query_Video_100_shot)

    # Query + Video -> Answer
    kg_qa_Query_Video = Dataset(
        dataset_name="kg_qa_Query_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/kg-llm/rephrased_QA_25Oct24_v2.2/training_SFT.json",
        image_path="./dataset/kg-llm/COIN/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(kg_qa_Query_Video)

    ## Directly use video
    # Query + Video -> Answer
    star_qa_Query_Video = Dataset(
        dataset_name="star_qa_Query_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/star/sft_annots_video_v3.3/STAR_train_NEAT_Query_Video_v3.3.json",
        image_path="./dataset/star/charadesv1_480/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(star_qa_Query_Video)

    ## Directly use video
    # Program + Video -> Answer
    star_qa_Program_Video = Dataset(
        dataset_name="star_qa_Program_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/star/sft_annots_video_v3.3/STAR_train_NEAT_Program_Video_v3.3.json",
        image_path="./dataset/star/charadesv1_480/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(star_qa_Program_Video)

    ## Directly use video
    # Program + Video -> Answer
    star_qa_Query_Program_Video = Dataset(
        dataset_name="star_qa_Query_Program_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/star/sft_annots_video_v3.3/STAR_train_NEAT_Query_Program_Video_v3.3.json",
        image_path="./dataset/star/charadesv1_480/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(star_qa_Query_Program_Video)

    ## Directly use video
    # Program + Video -> Answer
    star_qa_Query_Gen_Program_Video = Dataset(
        dataset_name="star_qa_Query_Gen_Program_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/star/sft_annots_video_v3.3/STAR_train_NEAT_Query_Gen_Program_Video_v3.3.json",
        image_path="./dataset/star/charadesv1_480/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(star_qa_Query_Gen_Program_Video)

    star_qa_Query_Gen_Program_3round_Video = Dataset(
        dataset_name="star_qa_Query_Gen_Program_3round_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/star/sft_annots_video_v3.3/STAR_train_NEAT_Query_Gen_Program_3round_Video_v3.3.json",
        image_path="./dataset/star/charadesv1_480/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(star_qa_Query_Gen_Program_3round_Video)

    star_qa_Query_Gen_Program_4round_Video = Dataset(
        dataset_name="star_qa_Query_Gen_Program_4round_Video",
        dataset_type="star_qa_decord",
        data_path="./dataset/star/sft_annots_video_v3.3/STAR_train_NEAT_Query_Gen_Program_4round_Video_v3.3.json",
        image_path="./dataset/star/charadesv1_480/video/",
        description="Original data source: https://bobbywu.com/STAR/",
    )
    add_dataset(star_qa_Query_Gen_Program_4round_Video)
