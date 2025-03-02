import os
import blingfire as bf
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams



class Translator:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        self.tokenizer = AutoTokenizer.from_pretrained('Unbabel/TowerInstruct-7B-v0.2')
        self.model = LLM("Unbabel/TowerInstruct-7B-v0.2")

        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=1024*2,
            temperature=0.0,
        )
    

    def process_prompt(self, text):
        messages = [
            {"role": "user", "content": f"Translate the following text from {self.src} into {self.tgt}.\n{self.src}: {text}\n{self.tgt}:"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        return self.tokenizer.decode(inputs)

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
        
        prompts = [self.process_prompt(x) for x in sentences]

        translated = self.model.generate(prompts, self.sampling_params)
        translated = [generations.outputs[0].text for generations in translated]

        return " ".join(translated)


if __name__ == '__main__':
    model = Translator('Portuguese', 'English')

    x = model("Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.")

    print(x)