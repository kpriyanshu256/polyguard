import os
import blingfire as bf
import torch

from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


from translation.flores_codes import flores_codes


class Translator:
    def __init__(self, src, tgt):
        self.src = flores_codes[src]
        try:
            self.tgt = flores_codes[tgt]
        except:
            self.tgt = tgt
            print("Setting target to ", tgt)

        model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B')
        tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-3.3B')

        self.pipeline = pipeline("translation", 
                    model=model, tokenizer=tokenizer, 
                    src_lang=self.src, tgt_lang=self.tgt, device=0)

    def __call__(self, text):
        
        if text == "" or text is None:
            return ""

        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(text)
        except:
            print("Text:", text)
            raise Exception("Bling error")

        sentences = []

        for o in sentence_offsets:
            sentences.append(text[o[0]:o[1]])
        
        translated = [
            self.pipeline(x, max_length=512)[0]['translation_text']
            for x in sentences
        ]

        return " ".join(translated)