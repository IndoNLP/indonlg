from argparse import ArgumentParser
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification, AlbertModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForPreTraining, BertModel
from transformers import XLMConfig, XLMTokenizer, XLMForSequenceClassification, XLMForTokenClassification, XLMModel
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel
from transformers import BartConfig, BartTokenizer, BartModel, BartForConditionalGeneration
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from modules.tokenization_indonlg import IndoNLGTokenizer
from modules.tokenization_mbart52 import MBart52Tokenizer

import json
import numpy as np
import torch

NLG_VOCAB_PATH = './vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model'
BART_CKPT_PATH = './checkpoints/IndoNLG_finals_mBart_model_checkpoint_6_36000.pt'

class WordSplitTokenizer():
    def tokenize(self, string):
        return string.split()
    
class SimpleTokenizer():
    def __init__(self, vocab, word_tokenizer, lower=True):
        self.vocab = vocab
        self.lower = lower
        idx = len(self.vocab.keys())
        self.vocab["<bos>"] = idx+0
        self.vocab["<|endoftext|>"] = idx+1
        self.vocab["<speaker1>"] = idx+2
        self.vocab["<speaker2>"] = idx+3
        self.vocab["<pad>"] = idx+4
        self.vocab["<cls>"] = idx+5
        self.vocab["<sep>"] = idx+6

        self.inverted_vocab = {int(v):k for k,v in self.vocab.items()}
        assert len(self.vocab.keys()) == len(self.inverted_vocab.keys())
        
        # Define word tokenizer
        self.tokenizer = word_tokenizer
        
        # Add special token attribute
        self.cls_token_id = self.vocab["<cls>"]
        self.sep_token_id = self.vocab["<sep>"]   

    def __len__(self):
        return len(self.vocab.keys())+1

    def convert_tokens_to_ids(self,tokens):
        if(type(tokens)==list):
            return [self.vocab[tok] for tok in tokens]
        else:
            return self.vocab[tokens]

    def encode(self,text,text_pair=None,add_special_tokens=False):
        if self.lower:
            text = text.lower()
            text_pair = text_pair.lower() if text_pair else None

        if not add_special_tokens:
            tokens = [self.vocab[tok] for tok in self.tokenizer.tokenize(text)]
            if text_pair:
                tokens += [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)]
        else:
            tokens = [self.vocab["<cls>"]] + [self.vocab[tok] for tok in self.tokenizer.tokenize(text)] + [self.vocab["<sep>"]]
            if text_pair:
                tokens += [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)] + [self.vocab["<sep>"]]
        return tokens     
    
    def encode_plus(self,text,text_pair=None,add_special_tokens=False, return_token_type_ids=False):
        if self.lower:
            text = text.lower()
            text_pair = text_pair.lower() if text_pair else None
        
        if not add_special_tokens:
            tokens = [self.vocab[tok] for tok in self.tokenizer.tokenize(text)]
            if text_pair:
                tokens_pair = [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)]
                token_type_ids = len(tokens) * [0] + len(tokens_pair) * [1]
                tokens += tokens_pair
        else:
            tokens = [self.vocab["<cls>"]] + [self.vocab[tok] for tok in self.tokenizer.tokenize(text)] + [self.vocab["<sep>"]]
            if text_pair:
                tokens_pair = [self.vocab[tok] for tok in self.tokenizer.tokenize(text_pair)] + [self.vocab["<sep>"]]
                token_type_ids = (len(tokens) * [0]) + (len(tokens_pair) * [1])
                tokens += tokens_pair
        
        encoded_inputs = {}
        encoded_inputs['input_ids'] = tokens
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = token_type_ids
        return encoded_inputs

    def decode(self,index,skip_special_tokens=True):
        return " ".join([self.inverted_vocab[ind] for ind in index])

    def save_pretrained(self, save_dir): 
        with open(save_dir+'/vocab.json', 'w') as fp:
            json.dump(self.vocab, fp, indent=4)

def gen_embeddings(vocab_list, emb_path, emb_dim=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = None
    count, pre_trained = 0, 0
    vocab_map = {}
    for i in range(len(vocab_list)):
        vocab_map[vocab_list[i]] = i

    found_word_map = {}

    print('Loading embedding file: %s' % emb_path)
    for line in open(emb_path).readlines():
        sp = line.split()
        count += 1
        if count == 1 and emb_dim is None:
            # header <num_vocab, emb_dim>
            emb_dim = int(sp[1])
            embeddings = np.random.rand(len(vocab_list), emb_dim)
            print('Embeddings: %d x %d' % (len(vocab_list), emb_dim))
        else:
            if count == 1:
                embeddings = np.random.rand(len(vocab_list), emb_dim)
                print('Embeddings: %d x %d' % (len(vocab_list), emb_dim))
                continue

            if(len(sp) == emb_dim + 1): 
                if sp[0] in vocab_map:
                    found_word_map[sp[0]] = True
                    embeddings[vocab_map[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print("Error:", sp[0], len(sp))
    pre_trained = len(found_word_map)
    print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / len(vocab_list)))
    return embeddings

def load_vocab(path):
    vocab_list = []
    with open(path, "r") as f:
        for word in f:
            vocab_list.append(word.replace('\n',''))

    vocab_map = {}
    for i in range(len(vocab_list)):
        vocab_map[vocab_list[i]] = i
        
    return vocab_list, vocab_map

def get_model_class(model_type, task):
    if 'babert-lite' in model_type:
        base_cls = AlbertModel
        if 'sequence_classification' == task:
            pred_cls = AlbertForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = AlbertForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = AlbertForMultiLabelClassification     
    elif 'xlm-mlm' in model_type:
        base_cls = XLMModel
        if 'sequence_classification' == task:
            pred_cls = XLMForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = XLMForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = XLMForMultiLabelClassification
    elif 'xlm-roberta' in model_type:
        base_cls = XLMRobertaModel
        if 'sequence_classification' == task:
            pred_cls = XLMRobertaForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = XLMRobertaForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = XLMRobertaForMultiLabelClassification
    else: # 'babert', 'bert-base-multilingual', 'word2vec', 'fasttext', 'scratch'
        base_cls = BertModel
        if 'sequence_classification' == task:
            pred_cls = BertForSequenceClassification
        elif 'token_classification' == task:
            pred_cls = BertForWordClassification
        elif 'multi_label_classification' == task:
            pred_cls = BertForMultiLabelClassification
    return base_cls, pred_cls

def load_word_embedding_model(model_type, task, vocab_path, word_tokenizer_class, emb_path, num_labels, lower=True):
    # Load config
    config = BertConfig.from_pretrained('bert-base-uncased') 

    # Init word tokenizer
    word_tokenizer = word_tokenizer_class()
    
    # Load vocab
    _, vocab_map = load_vocab(vocab_path)
    tokenizer = SimpleTokenizer(vocab_map, word_tokenizer, lower=lower)
    vocab_list = list(tokenizer.vocab.keys())

    # Adjust config
    if type(num_labels) == list:
        config.num_labels = max(num_labels)
        config.num_labels_list = num_labels
    else:
        config.num_labels = num_labels
    config.num_hidden_layers = num_labels
    
    if 'word2vec' in model_type:
        embeddings = gen_embeddings(vocab_list, emb_path)
        config.hidden_size = 400
        config.num_attention_heads = 8                                                        
    else: # 'fasttext'
        embeddings = gen_embeddings(vocab_list, emb_path, emb_dim=300)
        config.hidden_size = 300
        config.num_attention_heads = 10  
    config.vocab_size = len(embeddings)

    # Instantiate model
    if 'sequence_classification' == task:
        model = BertForSequenceClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
    elif 'token_classification' == task:
        model = BertForWordClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))
    elif 'multi_label_classification' == task:
        model = BertForMultiLabelClassification(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.FloatTensor(embeddings))        
    return model, tokenizer

def load_eval_model(args):
    vocab_path = f'./{args["model_dir"]}/{args["dataset"]}/{args["experiment_name"]}/vocab.txt'
    config_path = f'./{args["model_dir"]}/{args["dataset"]}/{args["experiment_name"]}/config.json'
    model_path = f'./{args["model_dir"]}/{args["dataset"]}/{args["experiment_name"]}/best_model_0.th'
    
    # Load for word2vec and fasttext
    if 'word2vec' in args['model_type'] or 'fasttext' in args['model_type']:
        emb_path = args['embedding_path'][args['model_type']]
        model, tokenizer = load_word_embedding_model(
            args['model_type'], args['task'], vocab_path, 
            args['word_tokenizer_class'], emb_path, args['num_labels'], lower=args['lower']
        )
        return model, tokenizer
        
    # Load config & tokenizer
    if 'albert' in args['model_type']:
        config = AlbertConfig.from_json_file(config_path)
        tokenizer = BertTokenizer(vocab_path)
    elif 'babert' in args['model_type']:
        config = BertConfig.from_json_file(config_path)
        tokenizer = BertTokenizer(vocab_path)
    elif 'scratch' in args['model_type']:
        config = BertConfig.from_pretrained('bert-base-uncased') 
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif 'bert-base-multilingual' in args['model_type']:
        config = BertConfig.from_pretrained(args['model_type'])
        tokenizer = BertTokenizer.from_pretrained(args['model_type'])
    elif 'xlm-mlm-100-1280' in args['model_type']:
        config = XLMConfig.from_pretrained(args['model_type'])
        tokenizer = XLMTokenizer.from_pretrained(args['model_type'])
    elif 'xlm-roberta' in args['model_type']:
        config = XLMRobertaConfig.from_pretrained(args['model_type'])
        tokenizer = XLMRobertaTokenizer.from_pretrained(args['model_type'])
    else:
        raise ValueError('Invalid `model_type` argument values')
    
    # Get model class
    base_cls, pred_cls = get_model_class(args['model_type'], args['task'])
        
    # Adjust config
    if type(args['num_labels']) == list:
        config.num_labels = max(args['num_labels'])
        config.num_labels_list = args['num_labels']
    else:
        config.num_labels = args['num_labels']    
        
    # Instantiate model
    model = pred_cls(config=config)
    base_model = base_cls.from_pretrained(model_path, from_tf=False, config=config)
    
    # Plug pretrained base model to classification model
    if 'bert' in model.__dir__():
        model.bert = base_model
    elif 'albert' in model.__dir__():
        model.albert = base_model
    elif 'roberta' in model.__dir__():
        model.roberta = base_model
    elif 'transformer' in model.__dir__():
        model.transformer = base_model
    else:
        ValueError('Model attribute not found, is there any change in the `transformers` library?')    
                                                        
    return model, tokenizer

def load_model(args, resize_embedding=True):
    if 'transformer' in args['model_type']:
        # baseline transformer models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
        config = BartConfig.from_pretrained('facebook/bart-base') # Use Bart config, because there is no MBart-base
        config.vocab_size = 40004
        
        # Instantiate model
        model = MBartForConditionalGeneration(config=config)
        if args['model_checkpoint']:
            bart = BartModel(config=config)
        
    elif 'baseline' in args['model_type']:
        vocab_path, config_path = None, None
        if 'mbart' in args['model_type']:
            # mbart models
            # Prepare config & tokenizer
            tokenizer = MBart52Tokenizer.from_pretrained(args['model_checkpoint'], src_lang=args['source_lang_bart'], tgt_lang=args['target_lang_bart'])
            model = MBartForConditionalGeneration.from_pretrained(args['model_checkpoint'])
            
            # Added new language token For MT
            if resize_embedding:
                model.resize_token_embeddings(model.config.vocab_size + 4) # For su_SU, jv_JV, <speaker_1>, <speaker_2>
            
            # Freeze Layer
            if args['freeze_encoder']:
                for parameter in model.model.encoder.parameters():
                    parameter.requires_grad = False
            if args['freeze_decoder']:
                for parameter in model.model.decoder.parameters():
                    parameter.requires_grad = False
            
        elif 'mt5' in args['model_type']:
            # mt5 models
            # Prepare config & tokenizer
            tokenizer = T5Tokenizer.from_pretrained(args['model_checkpoint'])
            model = MT5ForConditionalGeneration.from_pretrained(args['model_checkpoint'])   
            
            if 'small' not in args['model_type']:
                # Freeze Layer
                if args['freeze_encoder']:
                    for parameter in model.encoder.parameters():
                        parameter.requires_grad = False
                if args['freeze_decoder']:
                    for parameter in model.decoder.parameters():
                        parameter.requires_grad = False
    elif 'indo-bart' in args['model_type']:
        # bart models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
        config = BartConfig.from_pretrained('facebook/bart-base') # Use Bart config, because there is no MBart-base
        config.vocab_size = 40004
        
        # Instantiate model
        model = MBartForConditionalGeneration(config=config)
        if args['model_checkpoint']:
            bart = BartModel(config=config)
            bart.load_state_dict(torch.load(args['model_checkpoint'])['model'], strict=False)
            bart.shared.weight = bart.encoder.embed_tokens.weight
            model.model = bart
        
    elif 'indo-t5' in args['model_type']:
        # t5 models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
        config = T5Config.from_pretrained('t5-base')
        config.vocab_size = 40004
        
        # Instantiate model
        model = T5ForConditionalGeneration(config=config)
        if args['model_checkpoint']:
            model.load_state_dict(torch.load(args['model_checkpoint'])['model'], strict=False) # This has not been tested yet
        
    elif 'indo-gpt2' in args['model_type']:
        # gpt2 models
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = IndoNLGTokenizer(vocab_file=args['vocab_path'])
        config = GPT2Config.from_pretrained('gpt2')
        config.vocab_size = 40005
        
        # Instantiate model
        model = GPT2LMHeadModel(config=config)
        if args['model_checkpoint']:
            state_dict = torch.load(args['model_checkpoint'])
            model.load_state_dict(state_dict)
        
    return model, tokenizer, vocab_path, config_path
