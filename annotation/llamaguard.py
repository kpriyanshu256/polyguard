import os
import torch
import sys
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

unsafe_label2category = {
    'S1': 'S1: Violent Crimes',
    'S2': 'S2: Non-Violent Crimes',
    'S3': 'S3: Sex-Related Crimes',
    'S4': 'S4: Child Sexual Exploitation',
    'S5': 'S5: Defamation',
    'S6': 'S6: Specialized Advice',
    'S7': 'S7: Privacy',
    'S8': 'S8: Intellectual Property',
    'S9': 'S9: Indiscriminate Weapons',
    'S10': 'S10: Hate',
    'S11': 'S11: Suicide & Self-Harm',
    'S12': 'S12: Sexual Content',
    'S13': 'S13: Elections',
    'S14': 'S14: Code Interpreter Abuse'
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def moderate(chat, tokenizer, model, device):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def post_process(text):
    if 'unsafe' in text:
        category = text.split('unsafe')[1].strip()
        categories = category.split(',')
        if len(categories) == 1:
            return 'unsafe', unsafe_label2category[category]
        else:
            labels = [unsafe_label2category[c] for c in categories]
            return 'unsafe', ', '.join(labels)
    return 'safe', 'safe'

if __name__ == '__main__':

    DS, SPLIT = sys.argv[1:]
    df = load_dataset(DS, split=SPLIT).to_pandas()
    print(df)

    model_id = "meta-llama/Llama-Guard-3-8B"
    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map=device, 
        attn_implementation="flash_attention_2"
    )
    model.eval()
    
    prompt_labels = []
    prompt_categories = []
    response_labels = []
    response_categories = []
    
    for i, x in tqdm(df.iterrows(), total=len(df)):
        prompt = x['prompt']
        prompt_chat = [{"role": "user", "content": prompt}]
        if prompt is not None:
            prompt_prediction = moderate(prompt_chat, tokenizer, model, device)
            prompt_label, prompt_category = post_process(prompt_prediction)
        else:
            prompt_label, prompt_category = 'none', 'none'
        prompt_labels.append(prompt_label)
        prompt_categories.append(prompt_category)

        response = x['response']
        response_chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        if response is not None:
            response_prediction = moderate(response_chat, tokenizer, model, device)
            response_label, response_category = post_process(response_prediction)
        else:
            response_label, response_category = 'none', 'none'
        response_labels.append(response_label)
        response_categories.append(response_category)

    df['prompt_label'] = prompt_labels
    df['prompt_category'] = prompt_categories
    df['response_label'] = response_labels
    df['response_category'] = response_categories

    dataset = Dataset.from_pandas(df, split='train')
    dataset_name = f"{DS}-llama_guard"
    dataset.push_to_hub(dataset_name, private=True)