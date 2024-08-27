'''
Following code is adapted from
https://github.com/openai/CLIP
https://github.com/flairNLP/flair
'''

import os
from parser_tool import LanguageParser
import argparse
import unicodedata
import jsonlines
from tqdm import tqdm

# load tagger

def remove_accented_chars(text):
    text = text.replace("'", " ")
    normalized_text = unicodedata.normalize('NFD', text)
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('utf-8')
    return ascii_text

def get_attn_list(binding_ids, entity_ids):

    new_list = []
    for attrs, attr_positions in binding_ids:
        for e, e_position in entity_ids:
            # print(e_position)
            # print(attr_positions)
            if e_position[0] == attr_positions[-1]:
                entity = e
                new_list.append([entity, attrs, attr_positions, None])
            # else:
            #      entity_ids.remove((e, e_position))
                break

    used_positions = list(set([p for attrs, positions, _, _ in new_list for p in positions]))
    for e, e_position in entity_ids:
        if e[0] not in used_positions:
            new_list.append([e, [], e_position,  None])
    return new_list
            

def process_data(data, output_path, tokenizer):
    parser = LanguageParser(tokenizer_name_or_path=tokenizer)
    
    for entry in tqdm(data):
        caption = entry["query"]
        caption = remove_accented_chars(caption)
        # import pdb;pdb.set_trace()
        
        binding_ids, entity_ids = parser.extract_binding_indices(caption)
        # import pdb;pdb.set_trace()
        '''

        '''
        attn_list = get_attn_list(binding_ids, entity_ids)
        entry['attn_list'] = attn_list
        entry['Binding_relations'] = binding_ids
        entry['Entities'] = entity_ids
        
    output_dir = os.path.dirname(output_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path",default="data/anet/metadata/train.jsonl", type=str)
    parser.add_argument("--output_json_path", default="data/anet/metadata/train.jsonl",type=str)
    parser.add_argument("--tokenizer", default="clip-vit-base-patch32",type=str)
    args = parser.parse_args()

    input_json_path = args.input_json_path
    output_json_path = args.output_json_path
    
    input_json_data = []

    with jsonlines.open(input_json_path, 'r') as reader:
        input_json_data = list(reader)

    process_data(input_json_data,output_json_path,tokenizer=args.tokenizer)
