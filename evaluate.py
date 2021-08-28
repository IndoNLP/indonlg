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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
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
def evaluate(model, data_loader, forward_fn, metrics_fn, model_type, tokenizer, beam_size=1, max_seq_len=512, is_test=False, device='cpu', length_penalty=1.0):
    model.eval()
    torch.set_grad_enabled(False)
    
    total_loss, total_correct, total_labels = 0, 0, 0

    list_hyp, list_label = [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_fn(model, batch_data, model_type=model_type, tokenizer=tokenizer, device=device, is_inference=is_test, 
                                                      is_test=is_test, skip_special_tokens=True, beam_size=beam_size, max_seq_len=max_seq_len, length_penalty=length_penalty)
        
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
        return total_loss, metrics, list_hyp, list_label
    else:
        return total_loss, metrics

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()
    args = append_dataset_args(args)
    args = append_model_args(args)

    # create directory
    model_dir = '{}/{}/{}'.format(args["model_dir"],args["dataset"],args['experiment_name'])
    if not os.path.exists(model_dir):
        raise Exception(f'model directory `{model_dir}` not exists')

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
    
    metrics_scores = []
    result_dfs = []
    # load model
    model, tokenizer, vocab_path, config_path = load_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    if args['fp16']:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16'])

    if args['device'] == "cuda":
        model = model.cuda()

    if type(tokenizer) == IndoNLGTokenizer:
        src_lid = tokenizer.special_tokens_to_ids[args['source_lang']]
        tgt_lid = tokenizer.special_tokens_to_ids[args['target_lang']]
        
        # Inject lang id as bos token in `model.generate()` function
        tokenizer.bos_token = args['target_lang']
        model.config.decoder_start_token_id = tgt_lid
    elif type(tokenizer) == MBart52Tokenizer:
        src_lid = tokenizer.lang_code_to_id[args['source_lang_bart']]
        tgt_lid = tokenizer.lang_code_to_id[args['target_lang_bart']]  
        model.config.decoder_start_token_id = tgt_lid      
    elif type(tokenizer) == T5Tokenizer: # mT5 baseline goes here because it doesn't need any language token
        src_lid = -1
        tgt_lid = -1
        tokenizer.bos_token_id = tokenizer.decode([model.config.decoder_start_token_id])
    else:
        ValueError(f'Unknown tokenizer type `{type(tokenizer)}`')
        
    print("=========== TRAINING PHASE ===========")

    test_dataset_path = args['test_set_path']
    test_dataset = args['dataset_class'](test_dataset_path, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'], 
                                    speaker_1_id=args['speaker_1_id'], speaker_2_id=args['speaker_2_id'], separator_id=args['separator_id'],
                                    max_token_length=args['max_seq_len'], swap_source_target=args['swap_source_target'] if 'swap_source_target' in args else False)
    test_loader = args['dataloader_class'](dataset=test_dataset, model_type=args['model_type'], tokenizer=tokenizer, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=False)

    # Save Meta
    if vocab_path:
        shutil.copyfile(vocab_path, f'{model_dir}/vocab.txt')
    if config_path:
        shutil.copyfile(config_path, f'{model_dir}/config.json')
        
    # Load best model
    model.load_state_dict(torch.load(model_dir + "/best_model_0.th"))

    # Evaluate
    print("=========== EVALUATION PHASE ===========")
    test_loss, test_metrics, test_hyp, test_label = evaluate(model, data_loader=test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], 
            model_type=args['model_type'], tokenizer=tokenizer, beam_size=args['beam_size'], max_seq_len=args['max_seq_len'], is_test=True, device=args['device'], length_penalty=args['length_penalty'])

    metrics_scores.append(test_metrics)
    result_dfs.append(pd.DataFrame({
        'hyp': test_hyp, 
        'label': test_label
    }))
    
    result_df = pd.concat(result_dfs)
    metric_df = pd.DataFrame.from_records(metrics_scores)
    
    print('== Prediction Result ==')
    print(result_df.head())
    print()
    
    print('== Model Performance ==')
    print(metric_df.describe())

    result_df.to_csv(model_dir + "/prediction_result_latest_" + str(args["length_penalty"]) + ".csv")
    metric_df.describe().to_csv(model_dir + "/evaluation_result_latest_" + str(args["length_penalty"]) + ".csv")