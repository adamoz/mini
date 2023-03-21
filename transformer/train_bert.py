import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import bert_medium_config, bert_medium_finetune_config
from model import BERT, BERTCore, BERTFinetune
from utils import adjust_optimizer_lr
from data import Data, BERTDataset, BERTFinetuneDataset, DataMode
torch.manual_seed(1337)

train = False
finetune = True
bert_core_checkpoint = "./checkpoints/bert_medium_core_1_945_9500.pt"

train_config = bert_medium_config
finetune_config = bert_medium_finetune_config
data = Data(train_config, mode=DataMode.BERT)

if train:
    print("Training BERT")
    train_config = data.adjust_config(train_config)
    train_dataset = BERTDataset(data, split='train')
    valid_dataset = BERTDataset(data, split='valid')
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=train_config.batch_size)
    valid_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=train_config.batch_size)
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    bert_core = BERTCore(train_config)
    model = BERT(bert_core, train_config)
    model = model.to(train_config.device)
    optimizer = model.get_optimizer()
    print(train_config)

    best_ce = np.inf
    best_bert_core_checkpoint = None
    for step in range(train_config.max_iters):
        optimizer  = adjust_optimizer_lr(optimizer, step, train_config)

        if step % train_config.eval_interval == 0:
            losses = model.estimate_loss(train_loader, valid_loader)
            print(f"Step {step:<5} | train loss: {losses['train'][0]:<8.4f} | valid loss: {losses['valid'][0]:<8.4f} | train acc: {losses['train'][1]:<8.2f} | valid acc: {losses['valid'][1]:<8.2f}")
            if losses['valid'][0] < best_ce:
                best_ce = losses['valid'][0]
                best_bert_core_checkpoint = f'checkpoints/{train_config.name}_core_{str(round(losses["valid"][0].item(), 3)).replace(".", "_")}_{step}.pt'
                torch.save(model.bert_core.state_dict(), best_bert_core_checkpoint)
                torch.save(model.state_dict(), f'checkpoints/{train_config.name}_{str(round(losses["valid"][0].item(), 3)).replace(".", "_")}_{step}.pt')

        for micro_step in range(train_config.gradient_accumulation_steps):
            try:
                b = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                b = next(train_iter)

            x, segment_ids, y, does_continue = b['x'], b['segment_ids'], b['y'], b['does_continue']
            x, segment_ids, y, does_continue = x.to(train_config.device), segment_ids.to(train_config.device), y.to(train_config.device), does_continue.to(train_config.device)

            _, _, loss = model(x, segments=segment_ids, mask_sentence_target=y, next_sentece_target=does_continue)
            loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grand_norm_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

if finetune:
    print("Fine-tuning")
    bert_core_checkpoint = bert_core_checkpoint if bert_core_checkpoint else best_bert_core_checkpoint if train else None
    assert bert_core_checkpoint is not None, "No checkpoint provided"

    data.split_data(mode=DataMode.BERT_FINETUNE)
    finetune_config = data.adjust_config(finetune_config)

    train_dataset = BERTFinetuneDataset(data, split='train')
    valid_dataset = BERTFinetuneDataset(data, split='valid')
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=finetune_config.batch_size)
    valid_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=finetune_config.batch_size)
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    bert_core = BERTCore(finetune_config)
    bert_core.load_state_dict(torch.load(bert_core_checkpoint))    
    model = BERTFinetune(bert_core, finetune_config)
    model = model.to(finetune_config.device)
    optimizer = model.get_optimizer()
    print(finetune_config)

    best_ce = np.inf
    for step in range(finetune_config.max_iters):
        optimizer  = adjust_optimizer_lr(optimizer, step, finetune_config)

        if step % finetune_config.eval_interval == 0:
            losses = model.estimate_loss(train_loader, valid_loader)
            print(f"Step {step:<5} | train loss: {losses['train'][0]:<8.4f} | valid loss: {losses['valid'][0]:<8.4f} | train acc: {losses['train'][1]:<8.2f} | valid acc: {losses['valid'][1]:<8.2f}")
            print("Train accuracies")
            for idx, v in enumerate(losses['train'][2]):
                print(f"{data.decode_person(idx)}: {v / losses['train'][3][idx] * 100:.2f}")
            print("Valid accuracies")
            for idx, v in enumerate(losses['valid'][2]):
                print(f"{data.decode_person(idx)}: {v / losses['valid'][3][idx] * 100:.2f}")
            if losses['valid'][0] < best_ce:
                best_ce = losses['valid'][0]
                torch.save(model.bert_core.state_dict(), f'checkpoints/{finetune_config.name}_core_{str(round(losses["valid"][0].item(), 3)).replace(".", "_")}_{step}.pt')
                torch.save(model.state_dict(), f'checkpoints/{finetune_config.name}_{str(round(losses["valid"][0].item(), 3)).replace(".", "_")}_{step}.pt')

        for micro_step in range(finetune_config.gradient_accumulation_steps):
            try:
                b = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                b = next(train_iter)

            x, segment_ids, person_ids = b['x'], b['segment_ids'], b['person_id']
            x, segment_ids, person_ids = x.to(finetune_config.device), segment_ids.to(finetune_config.device), person_ids.to(finetune_config.device)

            _, loss = model(x, segments=segment_ids, target=person_ids)
            loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), finetune_config.grand_norm_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)