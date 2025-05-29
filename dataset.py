import random
import torch
from torch.utils.data import Dataset
import pandas as pd

class PersonaDataset(Dataset):
    def __init__(self, data, tokenizer, persona_list, max_seq_len=50):
        self.data = data  # pandas DataFrame
        self.tokenizer = tokenizer
        self.persona_list = persona_list
        self.max_seq_len = max_seq_len

        self.bos = tokenizer.bos_token_id or tokenizer.cls_token_id
        self.eos = tokenizer.eos_token_id or tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id
        self.ignore_index = -100
    
    def __len__(self):
        return len(self.data)
    
    def padding_ids(self, seq, pad_id):
        if len(seq) > self.max_seq_len:
            return seq[:self.max_seq_len]
        return seq + [pad_id] * (self.max_seq_len - len(seq))

    def __getitem__(self, i):
        persona_tag = f"<문어체>"
        data = self.data.iloc[i]
        for _ in range(15):
            persona = random.choice(self.persona_list.keys())
            target = data[persona]
            if not pd.isna(target):
                persona_tag = f"<{self.persona_list[persona]}>"
                break
        
        # 2. source sentence
        source = f"{persona_tag} {self.data.iloc[i]['formal']}"
        source_ids = [self.bos] + self.tokenizer(source, add_special_tokens=False)["input_ids"] + [self.eos]
        source_attention_mask = [1] * len(source_ids)

        # 3. target sentence
        target = str(self.data.iloc[i][persona])
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"] + [self.eos]

        input_ids = self.padding_ids(source_ids, self.pad)
        attention_mask = self.padding_ids(source_attention_mask, 0)
        labels = self.padding_ids(target_ids, self.ignore_index)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
