#!/usr/bin/env python3

import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def translate_file(file, translate):
    df = pd.read_csv(file, sep='\t', encoding='utf-8')
    da_to_translate = df['da'].values.tolist()
    df['en_mt'] = [e["translation_text"] for e in translate(da_to_translate, truncation=True)]
    df.to_csv(file, sep='\t', encoding='utf-8', index=False)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en")

    translate = pipeline("translation", model=model, tokenizer=tokenizer, batch_size=32, num_beams=4, device=0)
    translate_file("train.tsv", translate)
    translate_file("valid.tsv", translate)
    translate_file("test.tsv", translate)
