import os
import shutil
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW, T5Tokenizer
from nltk.tokenize import TweetTokenizer
from modules.tokenization_indonlg import IndoNLGTokenizer
from modules.tokenization_mbart52 import MBart52Tokenizer
from utils.functions import load_model
from utils.args_helper import get_parser, print_opts, append_dataset_args, append_model_args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

###
# Training & Evaluation Function
###
    
# Evaluate function for validation and test
def evaluate(model, data_loader, forward_fn, metrics_fn, model_type, tokenizer, beam_size=1, max_seq_len=512, is_test=False, device='cpu'):
    model.eval()
    torch.set_grad_enabled(False)
    
    total_loss, total_correct, total_labels = 0, 0, 0

    list_hyp, list_label = [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, device=device, is_inference=is_test, 
                                                      is_test=is_test, skip_special_tokens=True, beam_size=beam_size, max_seq_len=max_seq_len)
        
        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label

        if not is_test:
            # Calculate total loss for validation
            test_loss = loss.item()
            total_loss = total_loss + test_loss

            # pbar.set_description("VALID {}".format(metrics_to_string(metrics)))
            pbar.set_description("VALID LOSS:{:.4f}".format(total_loss/(i+1)))
        else:
            pbar.set_description("TESTING... ")
            # pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
    
    metrics = metrics_fn(list_hyp, list_label)        
    if is_test:
        return total_loss/(i+1), metrics, list_hyp, list_label
    else:
        return total_loss/(i+1), metrics

# Training function and trainer
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, tokenizer, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, max_norm=10, grad_accum=1, beam_size=1, max_seq_len=512, model_type='bart', model_dir="", exp_id=None, fp16=False, device='cpu'):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0

    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)
        
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            if fp16:
                with torch.cuda.amp.autocast():
                    loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, 
                                                device=device, skip_special_tokens=False, is_test=False)
                    
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()                    
            else:
                loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, 
                                            device=device, skip_special_tokens=False, is_test=False)
            
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(optimizer)))
            
            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                                   
        metrics = metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))
        
        # Decay Learning Rate
        scheduler.step()

        # evaluate
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate(model, valid_loader, forward_fn, metrics_fn, model_type, tokenizer, is_test=False, 
                                                 beam_size=beam_size, max_seq_len=max_seq_len, device=device)

            print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1), val_loss, metrics_to_string(val_metrics)))            
            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # save model
                if exp_id is not None:
                    torch.save(model.state_dict(), model_dir + "/best_model_" + str(exp_id) + ".th")
                else:
                    torch.save(model.state_dict(), model_dir + "/best_model.th")
                count_stop = 0
            else:
                count_stop += 1
                print("count stop:", count_stop)
                if count_stop == early_stop:
                    break
