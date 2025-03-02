import os
import sys
import json
import time
import pandas as pd
import litellm
from litellm import completion 
from litellm.utils import trim_messages
from pydantic import BaseModel, Field
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm
from enum import Enum

pd.set_option("display.max_colwidth", None)

system_prompt = """You are a linguistic expert. Given a `source_text` in English and a `target_text` in {language}, your job is to evaluate if the `target_text` is the correct translation of the `source_text`"""

user_msg = """`source_text`: {src}
`target_text`: {tgt}"""


os.environ["AZURE_API_KEY"] = ""
os.environ['AZURE_API_BASE'] = ""
os.environ["AZURE_API_VERSION"] = "" 
MODEL = 'azure/gpt-4o'

num_retries = 3


class QualityEnum(str, Enum):
    incorrect = 'translation is bad'
    almost_correct = 'translation is almost correct with a few errors'
    entirely_correct = 'translation is entirely correct'


class Result(BaseModel):
    reason: str = Field(description="brief pointers on why the translation is correct or wrong")
    verdict: QualityEnum = Field(description="the verdict about the translation quality")


def verify_translation(src, tgt, lang):
    messages = [
        {"role": "system", "content": system_prompt.format(language=lang)}, 
        {"role": "user", "content": user_msg.format(src=src, tgt=tgt)}
    ]

    while True:
        try:
            response = completion(
                model = MODEL,
                messages = trim_messages(messages),
                response_format = Result,
                temperature=0.0,
            )
            break
        except Exception as e:
            print("Retrying after 1 min | ", e)

            if 'encountered an issue with repetitive patterns' in str(e):
                response = {'choice': [{'message': {'content': "{}"}}]}
                break

            time.sleep(60)

    out = (response['choices'][0]['message']['content'])
    print(out)
    # print(json.loads(out))
    return out


    

    

if __name__ == '__main__':
    LANG = sys.argv[1]
    

    ds_orig = load_dataset("ToxicityPrompts/wildguard-test")
    print(ds_orig)

    ds_translated_name = f"ToxicityPrompts/tower-v2-wildguard-test-wildguardtest-{LANG}-0-1725"

    print("Translated data: ", ds_translated_name)
    ds = load_dataset(ds_translated_name)
    print(ds)

    ds_name = f'ToxicityPrompts/gpt4o-translation-checker-tower-v2-{LANG}'

    print("Saving to ", ds_name)

    ds = ds['test']
    df = ds.to_pandas()

    ds_orig = ds_orig['test']
    df_orig = ds_orig.to_pandas()

    df = pd.merge(df_orig, df, on='id')


    df['prompt_verdict'] = ""
    df['response_verdict'] = ""

    
    for i, r in tqdm(df.iterrows(), total=len(df)):
        src_prompt = r['prompt_x']
        tgt_prompt = r['prompt_y']

        src_response = r['response_x']
        tgt_response = r['response_y']

        prompt_verdict = verify_translation(src_prompt, tgt_prompt, LANG)


        if src_response is not None or len(src_response) > 0:
            response_verdict = verify_translation(src_response, tgt_response, LANG)
        
        df.loc[i, 'prompt_verdict'] = prompt_verdict
        df.loc[i, 'response_verdict'] = response_verdict



    ds_dict = {}
    ds_dict['test'] = Dataset.from_pandas(df)

    ds_dict = DatasetDict(ds_dict)
    print(ds_dict)
    ds_dict.push_to_hub(ds_name, private=True)