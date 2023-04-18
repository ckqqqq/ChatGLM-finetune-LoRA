
import os
import time
import tqdm
import json
import torch
import numpy as np
import loralib as lora
from lora_utils.insert_lora import get_lora_model


from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
# from accelerate import Accelerator, DeepSpeedPlugin
import accelerate
from transformers import get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader



checkpoint = "THUDM/chatglm-6b"

model_id = "finetune_test"

mixed_precision = 'bf16'
lora_config = {
    'r': 32,
    'lora_alpha':32,
    'lora_dropout':0.05,
    'enable_lora':[True, False, True],
}

LR = 1e-4
BATCH_SIZE = 1 #两边都会乘上
MAX_LENGTH = 256
NUM_EPOCHS = 3
accumulate_step = 4 # 这个数好像 
warm_up_ratio = 0.1



deepspeed_plugin = accelerate.DeepSpeedPlugin(gradient_accumulation_steps=accumulate_step)
accelerator = accelerate.Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin, log_with="tensorboard", project_dir='runs/')
device = accelerator.device


with accelerator.main_process_first():
    retry_cnt = 10
    cnt = 0
    while cnt < retry_cnt:
        try:
            import dataset.GLM
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
            model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
            if mixed_precision == None:
                model = model.float()
            break
        except:
            cnt += 1 

    model = get_lora_model(model, lora_config)


accelerator.wait_for_everyone()

model.use_cache = False
model.gradient_checkpointing = False


import dataset.Alpaca as Alpaca_Data
dataset.GLM.device = device



from transformers import AutoConfig
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision = 'main')
# config 是直接载入的
# pad_to = 1
# device = ' ckqoisjfgaosijgf'# 会在父文件赋值
eos_id = config.eos_token_id
pad_token_id = config.pad_token_id
pad_to = 1
def collate_fn(batch):
    global device
    input_ids = []
    labels = []
    position_ids = []
    # padding 其实如果为1的话就不需要
    
    _max_length = max([len(obj['prompt'])+len(obj['completion']) for obj in batch])
    # _max_length = (_max_length // pad_to + (_max_length % pad_to > 0) ) * pad_to

    attention_mask = torch.ones((len(batch), _max_length, _max_length), device=device)
    attention_mask.tril_()

    for i, obj in enumerate(batch):
        context_length = obj['prompt'].index(130004)
        attention_mask[i, :, :context_length] = 1

        to_pad = _max_length - len(obj['prompt']) - len(obj['completion'])

        input_ids.append(obj['prompt'] + obj['completion'] + [pad_token_id] * to_pad)

        position_ids.append(torch.stack([torch.arange(0, _max_length, device=device), 
                                         torch.concat([torch.zeros(context_length - 1, device=device), 
                                                       torch.arange(0, _max_length - context_length + 1, device=device)])]).long())

        labels.append(torch.tensor([-100] * len(obj['prompt']) + 
                                   obj['completion'] +
                                   [-100] * to_pad, device=device).long())

    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': attention_mask, 
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}
max_memory=accelerate.utils.get_max_memory()
accelerator.print("max_memory",max_memory,f"device= {device} ，{accelerator.device}")
with accelerator.main_process_first():
    accelerator.print('Start to process data')
    pairs = Alpaca_Data.load('./data/alpaca_data.json')
    pairs_encoded = dataset.GLM.encode_pairs(pairs, tokenizer)
    pairs_encoded = list(filter(lambda pair: len(pair['prompt'])+len(pair['completion']) <= MAX_LENGTH, pairs_encoded))
train_dataset = dataset.GLM.SimpleDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, shuffle=True, batch_size=BATCH_SIZE)



accelerator.wait_for_everyone()



optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(len(train_dataloader) // accumulate_step * NUM_EPOCHS),
)
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)



accelerator.print('Start to train')
accelerator.init_trackers(model_id, {})

total_effective_step = 0

for epoch in range(NUM_EPOCHS):

    batch_loss = 0
    effective_step = 0
    
    for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):

        outputs = model(**batch)

        loss_d = outputs.loss.detach().cpu().float().item()
        batch_loss += loss_d

        loss = outputs.loss / accumulate_step
        accelerator.backward(loss)

        if (step+1) % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            effective_step += 1

            gathered_batch_loss = accelerator.gather((torch.tensor(batch_loss, device=device)))

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "train_loss": gathered_batch_loss.mean().item() / accumulate_step,
                        "epoch": epoch,
                    },
                    step = total_effective_step + effective_step,
                )

            t.set_description(f"loss: {gathered_batch_loss.mean().item() / accumulate_step}")
            batch_loss = 0   
        
    
    accelerator.wait_for_everyone()
    
    total_effective_step += effective_step
    
    if accelerator.is_main_process:
        os.makedirs(f'saved/{model_id}', exist_ok = True)
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), f'saved/{model_id}/{model_id}_epoch_{epoch}.pt')

    accelerator.wait_for_everyone()

