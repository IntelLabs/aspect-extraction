import torch

class AddFineTunedPreds:

    def __init__(self, ace_preds) -> None:
        self.ace_preds = ace_preds

    def add_ace_preds(self, ex):

        all_text, all_tokens, all_tags, ace_preds = [], [], [], []
        for text, tokens, tags, ace_pred in zip(ex['text'], ex['tokens'], ex['tags'], self.ace_preds):
            all_text.append(text)
            all_tokens.append(tokens)
            all_tags.append(tags)
            ace_preds.append(ace_pred)

        return dict(text=all_text, tokens=all_tokens, tags=all_tags, ace_preds=ace_preds)