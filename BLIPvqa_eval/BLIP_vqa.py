import argparse
import os

import torch

from tqdm import tqdm, trange


import json
from tqdm.auto import tqdm
import sys
import spacy

from BLIP.train_vqa_func import VQA_main

def Create_annotation_for_BLIP(image_folder, outpath, np_index=None):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))#sort


    cnt=0

    #output annotation.json
    for file_name in file_names:
        image_dict={}
        # --- FIX: Use os.path.join to correctly create the file path ---
        image_dict['image'] = os.path.join(image_folder, file_name)
        image_dict['question_id']= cnt
        f = file_name.split('_')[0]
        doc = nlp(f)
        
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in ['top', 'the side', 'the left', 'the right']:  # todo remove some phrases
                noun_phrases.append(chunk.text)
        if(len(noun_phrases)>np_index):
            q_tmp = noun_phrases[np_index]
            image_dict['question']=f'{q_tmp}?'
        else:
            image_dict['question'] = ''
            

        image_dict['dataset']="color"
        cnt+=1

        annotations.append(image_dict)

    print('Number of Processed Images:', len(annotations))

    json_file = json.dumps(annotations)
    with open(f'{outpath}/vqa_test.json', 'w') as f:
        f.write(json_file)

def parse_args():
    parser = argparse.ArgumentParser(description="BLIP vqa evaluation.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        required=True,
        help="Path to output BLIP vqa score",
    )
    parser.add_argument(
        "--np_num",
        type=int,
        default=8,
        help="Noun phrase number, can be greater or equal to the actual noun phrase number",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np_index = args.np_num #how many noun phrases

    answer = []
    image_folder = os.path.join(args.out_dir, "samples")
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))
    
    sample_num = len(file_names)
    reward = torch.zeros((sample_num, np_index)).to(device='cuda')

    out_dir = args.out_dir
    order = "_blip" #rename file
    
    for i in tqdm(range(np_index), desc="Processing Noun Phrases"):
        print(f"start VQA{i+1}/{np_index}!")
        annotation_path = f"{out_dir}/annotation{i + 1}{order}"
        os.makedirs(os.path.join(annotation_path, "VQA", "result"), exist_ok=True)
        
        Create_annotation_for_BLIP(
            image_folder,
            annotation_path,
            np_index=i,
        )
        
        answer_tmp = VQA_main(f"{annotation_path}/", f"{annotation_path}/VQA/")
        answer.append(answer_tmp)

        with open(f"{annotation_path}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        with open(f"{annotation_path}/vqa_test.json", "r") as file:
            r_tmp = json.load(file)
        
        for k in range(len(r)):
            if r_tmp[k]['question']: # Check if question is not empty
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i+1}/{np_index}!")

    # Calculate final reward by multiplying scores for each noun phrase
    reward_final = reward[:,0]
    for i in range(1, np_index):
        reward_final *= reward[:,i]

    # --- MODIFIED BLOCK: Create and save the final results ---
    print("Generating final results file...")
    final_results = []
    total_score = 0
    
    for k in range(sample_num):
        # Create a dictionary with the image path and its final score
        result_dict = {
            'image_path': os.path.join(image_folder, file_names[k]),
            'answer': '{:.4f}'.format(reward_final[k].item())
        }
        final_results.append(result_dict)
        total_score += reward_final[k].item()

    # Create the final output directory and save the JSON file
    final_output_dir = f"{out_dir}/annotation{order}"
    os.makedirs(final_output_dir, exist_ok=True)
    with open(f"{final_output_dir}/vqa_result.json", "w") as file:
        json.dump(final_results, file, indent=4)
    # --- END OF MODIFIED BLOCK ---

    # Calculate and print the average score
    avg_score = total_score / len(final_results) if final_results else 0
    print("BLIP-VQA score:", avg_score, '!\n')
    with open(f"{final_output_dir}/blip_vqa_score.txt", "w") as file:
        file.write("BLIP-VQA score:" + str(avg_score))



if __name__ == "__main__":
    main()