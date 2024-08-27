import spacy
from transformers import AutoTokenizer
from .util import (extract_attribution_indices, 
                  extract_attribution_indices_with_verb_root, 
                  extract_attribution_indices_with_verbs,
                  extract_entities_only,
                  unify_lists,
                  get_indices,
                  align_wordpieces_indices)



class LanguageParser:
    def __init__(self, model_name: str = "en_core_web_trf",tokenizer_name_or_path: str = "clip-vit-base-patch32", include_entities: bool = True):
        print(f'sPacy: Using {model_name} model')
        self.parser = spacy.load(model_name)
        print(f'Using tokenizer form {tokenizer_name_or_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name_or_path)
        self.include_entities = include_entities
        self.start_token = self.tokenizer.bos_token
        self.end_token = self.tokenizer.eos_token

    def parse_prompt(self, prompt: str):
        return self.parser(prompt)

    def extract_binding_indices(self, prompt):
        # import pdb;pdb.set_trace()
        doc = self.parse_prompt(prompt)
        indices = []
        modifier_sets = []
        # extract standard attribution indices
        modifier_sets_1 = extract_attribution_indices(doc)
        indices_1 = self._align_indices(prompt, modifier_sets_1)
        if indices_1:
            indices.append(indices_1)
            modifier_sets.append(modifier_sets_1)
        # import pdb;pdb.set_trace()
        # extract attribution indices with verbs in between
        modifier_sets_2 = extract_attribution_indices_with_verb_root(doc)
        indices_2 = self._align_indices(prompt, modifier_sets_2)
        if indices_2:
            indices.append(indices_2)
            modifier_sets.append(modifier_sets_2)

        modifier_sets_3 = extract_attribution_indices_with_verbs(doc)
        # print(modifier_sets_3)
        indices_3 = self._align_indices(prompt, modifier_sets_3)
        if indices_3:
            indices.append(indices_3)
            modifier_sets.append(modifier_sets_3)

        modifier_sets = unify_lists(modifier_sets)
        indices_all = self._align_indices(prompt, modifier_sets)
        modifier_sets = [[token.text for token in tokens] for tokens in modifier_sets]
        binding_relations = list(zip(modifier_sets,indices_all))
        # list(zip(test,unify_lists(test1)))
        # entities only
        if self.include_entities:
            modifier_sets_4 = extract_entities_only(doc)
            indices_4 = self._align_indices(prompt, modifier_sets_4)
            modifier_sets_4 = unify_lists([modifier_sets_4])
            indices_4 = self._align_indices(prompt, modifier_sets_4)
            modifier_sets_4 = [[token.text for token in tokens] for tokens in modifier_sets_4]
            entities = list(zip(modifier_sets_4,indices_4))

            return binding_relations,entities
        else:
            # make sure there are no duplicates
            return binding_relations, None
        
    # def extract_vertical_relationships(self, prompt: str):
    #     # TODO
    #     doc = self.parse_prompt(prompt)
    #     vertical_relations = []
        
    #     for token in doc:
    #         # 检测到可能表示垂直方位关系的介词
    #         if token.dep_ == 'prep' and token.lemma_ in ['above', 'below', 'beneath', 'underneath', 'under', 'atop', 'ontop', 'overhead']:
    #             prep = token
    #             head = prep.head  # 获取介词的主词
    #             for child in prep.children:
    #                 # 介词宾语通常表示与主词相关的垂直空间位置
    #                 if child.dep_ == 'pobj':
    #                     # 检测上下关系
    #                     if prep.lemma_ in ['above', 'over']:
    #                         relation = ['above', 'below']
    #                     elif prep.lemma_ in ['below', 'under']:
    #                         relation = ['below', 'above']
    #                     vertical_relations.append([head.i, child.i, relation])

    #     return vertical_relations

    
    def _align_indices(self, prompt, spacy_pairs):
        # This function aligns word piece indices with spacy token pairs from the doc.
        # Detailed implementation is needed based on actual alignment logic.
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [self.start_token, self.end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text.lower() == wp.lower():
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.lower().startswith(wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                                [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            if curr_collected_wp_indices:
                paired_indices.append(curr_collected_wp_indices)
            else:
                print(f"No wordpieces were aligned for {pair} in _align_indices")

        return paired_indices



if __name__ == "__main__":
    prompt = 'There are seven people which is young and beautiful, standing on the huge mountain.'
    print(prompt)
    parser = LanguageParser()
    # doc = parser.parse_prompt(prompt)
    binding_ids, entity_ids = parser.extract_binding_indices(prompt)
    # print(doc)
    print(f"Binding relations: {binding_ids}")
    print(f"Entitied: {entity_ids}")

    promptb = 'The blue dog is under the table, while the bird in above the tree'
    print(promptb)
    # doc1 = parser.parse_prompt(promptb)
    binding_ids1, entity_ids1 = parser.extract_binding_indices(promptb)
    spatial_relations1 = parser.extract_vertical_relationships(promptb)
    # print(doc1)
    print(f"Binding relations: {binding_ids1}")
    print(f"Entitied: {entity_ids1}")