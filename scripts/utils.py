import re
import os
from datasets import load_dataset, load_from_disk
import pandas as pd


### System variables ###
EXPLANATION_TYPES = ["whyexp", "whynot", "implicit"]
DATASET_PREFIX = "azaninello/"
DATASET_NAME = "e-rte-3-it"
ORIGINAL_DATASET = DATASET_PREFIX + DATASET_NAME
GENERATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = f'''/home/azaninello/geese4CALAMITA/output/meta-llama__Meta-Llama-3-8B-Instruct''' #/{GENERATION_MODEL.replace("/", "__")}'''
PREDICTION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

LIMIT = 10

### Explanation generation ###

def generate_whyexp_explanation(x):
    prompt = f'''Your task is to clarify the entailment relationship between a pair of sentences by explaining why a classifier predicted a specific entailment label.\nSentence 1: {x["text_t"].strip()}\nSentence 2: {x["text_h"].strip()}\nEntailment relation label: {x["label"]} ({x["text_label"]}).\nExplain why label {x["label"]} is correct.'''
    return prompt

def generate_whynot_explanation(x):
    labels = ["YES", "NO", "UNKNOWN"]
    labels.remove(x["label"])
    prompt = f'''Your task is to clarify the entailment relationship between a pair of sentences by explaining why a classifier did not predict any other entailment label.\nSentence 1: {x["text_t"].strip()}\nSentence 2: {x["text_h"].strip()}\nEntailment relation label: {x["label"]} ({x["text_label"]}).\nExplain why labels {labels[0]} and {labels[1]} are not correct.'''
    return prompt

def generate_implicit_explanation(x):
    prompt = f'''Your task is to explain the entailment relationship between a pair of sentences by providing the implicit information connecting them.\nSentence 1: {x["text_t"].strip()}\nSentence 2: {x["text_h"].strip()}\nEntailment relation label: {x["label"]} ({x["text_label"]}).\nExplain how the two sentences are connected.'''
    return prompt

### Anonymization of explanations ###

def anonimyze_exp(text):
    # the anon_pattern needs to be customized according to task, labels and language
    anon_pattern = r"(\bYES\b|\bNO\b|\bUNKNOWN\b|\bentail\w*\b|\bcontradict\w*\b|\bneutral\w*\b|\bimpl\w*\b|\bcontradd\w*\b)"
    subst_str = "XXX"
    text = re.sub(anon_pattern, subst_str, text, flags=re.IGNORECASE)
    return text

### Create explained dataset ###

def get_generated_files(orig=ORIGINAL_DATASET, output_dir=OUTPUT_DIR, exp_types=EXPLANATION_TYPES):
    orig_dataset = load_dataset(orig, split='test')

    for exp_type in exp_types:
        for file in os.listdir(output_dir):
            task = f"geese_generation_{exp_type}"
            if re.match(fr"^samples_{task}_.*\.jsonl$", file):
                explained_dataset = load_dataset('json', data_files=os.path.join(output_dir, file), split='train')
                gen_exp_map = {item['doc']['id']: item['resps'][0][0].strip() for item in explained_dataset}
                
                def transform_example(example, exp_type=exp_type):
                    # Add 'anon_human' field by anonymizing the 'explanation' field
                    example['anon_human'] = anonimyze_exp(example['explanation']).strip()

                    # Add fields based on 'exp_type' and handle missing keys
                    #for exp_type in exp_types:  # exp_types should be defined as ["why", "whynot", "implicit"]
                    example[exp_type] = gen_exp_map.get(example['id'], '')  # Use .get() to avoid KeyError
                    example[f'anon_{exp_type}'] = anonimyze_exp(example[exp_type]).strip()

                    return example
                
                orig_dataset = orig_dataset.map(transform_example)

    # Convert to Pandas and save as JSONL
    df = orig_dataset.to_pandas()
    data_path = f"data/explained-{DATASET_NAME}-new.jsonl"
    os.makedirs("data", exist_ok=True)
    df.to_json(data_path, orient="records", lines=True, force_ascii=False)
    print(f"Written {data_path}")
    return data_path

#get_generated_files()
### Prediction with explanation ###

def predict_with_no_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: Not given.
Entailment label:'''
    return prompt

def predict_with_dummy_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: {x["text_h"]}.
Entailment label:'''
    return prompt

def predict_with_obvious_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: {x["label"]}.
Entailment label:'''
    return prompt

def predict_with_human_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: {x["anon_human"]}.
Entailment label:'''
    return prompt

def predict_with_why_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: {x["anon_whyexp"]}.
Entailment label:'''
    return prompt

def predict_with_whynot_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: {x["anon_whynot"]}.
Entailment label:'''
    return prompt

def predict_with_implicit_explanation(x):
    prompt = f'''You task in to predict the entailment relationship between 2 sentences chosing one label among "YES" (entailment), "NO" (contradiction), and "UNKNOWN" (neutrality).
Sentence 1: {x["text_t"]}\nSentence 2: {x["text_h"]}\nHint: {x["anon_implicit"]}.
Entailment label:'''
    return prompt