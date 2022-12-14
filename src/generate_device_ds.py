import json

# Example entry:
# {"tokens": ["Loaded", "with", "bloatware", "."], "tags": ["O", "O", "B-ASP", "O"], "text": "Loaded with bloatware."}

def remove_extra_spaces(text):
    t = text
    for p in "'.!?,":
        t = t.replace(f" {p}", p)
    t = t.replace(" n't", "n't")
    t = t.replace("( ", "(")
    t = t.replace(" )", ")")
    return t

all_examples = []

for split in 'test', 'train':
    with open(f'data/device/device_{split}.txt') as f_texts:
        with open(f'data/device/{split}.txt') as f_labels:
            labeled_sents = [s for s in f_labels.read().split('\n\n') if s]
            text_sents = [line.split('####')[0] for line in f_texts]

            assert len(labeled_sents) == len(text_sents)

            for labeled_sent, text in zip(labeled_sents, text_sents):
                tokens, tags = [], []
                for tok_tag_pair in labeled_sent.split('\n'):
                    tok, tag = tok_tag_pair.split('\t')
                    tokens.append(tok)
                    tags.append(tag)
                    fixed_text = remove_extra_spaces(text)

                all_examples.append(dict(tokens=tokens, tags=tags, text=fixed_text))

with open('data/device.json', 'w') as f_out:
    f_out.write('\n'.join([json.dumps(ex) for ex in all_examples]))