import os

import re
import sys
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translation.translator_tower import Translator



if __name__ == '__main__':

    DS, LANG, CONFIG, SPLIT, START, BATCH = sys.argv[1:]
    
    START = int(START)
    BATCH = int(BATCH)
    
    ds = load_dataset(DS, CONFIG)
    print(ds)

    model = Translator("English", LANG)

    def translate(x):
        x['prompt'] = model(x['prompt'])
        x['response'] = model(x['response'])

        return x

    ds_dict = {}

    ds = ds[SPLIT]

    END = min(START + BATCH, len(ds))
    ds_name = f'tower-v2-{DS.split("/")[1]}-{CONFIG}-{LANG}-{START}-{END}'
    print(f'Dataset name: ', ds_name)


    print(f'Translating {CONFIG} {SPLIT} to {LANG} {START}-{END}')


    ds = ds.select([i for i in range(START, END)])
    df = ds.to_pandas()


    for i, x in tqdm(df.iterrows(), total=len(df)):
        df.loc[i] = translate(x)

    ds_dict[SPLIT] = Dataset.from_pandas(df)

    ds_dict = DatasetDict(ds_dict)
    ds_dict.push_to_hub(ds_name, private=True)