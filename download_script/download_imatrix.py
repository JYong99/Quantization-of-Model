from datasets import load_dataset

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
with open('wiki.train.raw', 'w', encoding='utf-8') as f:
    for text in dataset['train']['text']:
        f.write(text + '\n')