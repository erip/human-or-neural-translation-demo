#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split

def read_lines(file):
    with open(file, encoding="utf-8") as f:
        return [line.strip() for line in f]

if __name__ == "__main__":
    data = { "en": read_lines("europarl-v7.da-en.en"), "da": read_lines("europarl-v7.da-en.da") }
    df = pd.DataFrame(data).dropna()
    df = df[(df['da'] != "") & (df['en'] != "")]
    train, dev_test = train_test_split(df, train_size=25_000, test_size=10_000, random_state=1234, shuffle=True)
    dev, test = train_test_split(dev_test, train_size=5_000, test_size=5_000, random_state=1234, shuffle=True)
    train.to_csv('train.tsv', encoding='utf-8', sep='\t', index=False)
    dev.to_csv('valid.tsv', encoding='utf-8', sep='\t', index=False)
    test.to_csv('test.tsv', encoding='utf-8', sep='\t', index=False)
