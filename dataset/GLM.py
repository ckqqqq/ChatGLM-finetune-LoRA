import torch
from transformers import AutoConfig
from torch.utils.data import Dataset

config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision = 'main')
# config 是直接载入的
pad_to = 1
device = ' ckqoisjfgaosijgf'# 会在父文件赋值
eos_id = config.eos_token_id
pad_token_id = config.pad_token_id

class SimpleDataset(Dataset):
    def __init__(self, pairs) -> None:
        super().__init__()
        self.pairs = pairs
 
    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)





def encode_pairs(pairs, tokenizer, with_eos = True):
    prompt_ids = tokenizer.batch_encode_plus([pair['prompt'] for pair in pairs])['input_ids']
    completion_ids = tokenizer.batch_encode_plus([pair['completion'] for pair in pairs], add_special_tokens=False)['input_ids']
    if with_eos:
        pairs_encoded = [{'prompt':prompt_ids[i], 'completion':completion_ids[i] + [eos_id]} for i in range(len(pairs))]
    else:
        pairs_encoded = [{'prompt':prompt_ids[i], 'completion':completion_ids[i]} for i in range(len(pairs))]
    return pairs_encoded