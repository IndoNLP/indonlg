import json
import numpy as np
import pandas as pd
import string
import torch
import re
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

#####
# Term Extraction Airy
#####
class AspectExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-SENTIMENT': 0, 'O': 1, 'I-ASPECT': 2, 'B-SENTIMENT': 3, 'B-ASPECT': 4}
    INDEX2LABEL = {0: 'I-SENTIMENT', 1: 'O', 2: 'I-ASPECT', 3: 'B-SENTIMENT', 4: 'B-ASPECT'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if '\t' in line:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
       
class AspectExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(AspectExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]
            
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# Ner Grit + Prosa
#####
class NerGritDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PERSON': 0, 'B-ORGANISATION': 1, 'I-ORGANISATION': 2, 'B-PLACE': 3, 'I-PLACE': 4, 'O': 5, 'B-PERSON': 6}
    INDEX2LABEL = {0: 'I-PERSON', 1: 'B-ORGANISATION', 2: 'I-ORGANISATION', 3: 'B-PLACE', 4: 'I-PLACE', 5: 'O', 6: 'B-PERSON'}
    NUM_LABELS = 7
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data) 

class NerProsaDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PPL': 0, 'B-EVT': 1, 'B-PLC': 2, 'I-IND': 3, 'B-IND': 4, 'B-FNB': 5, 'I-EVT': 6, 'B-PPL': 7, 'I-PLC': 8, 'O': 9, 'I-FNB': 10}
    INDEX2LABEL = {0: 'I-PPL', 1: 'B-EVT', 2: 'B-PLC', 3: 'I-IND', 4: 'B-IND', 5: 'B-FNB', 6: 'I-EVT', 7: 'B-PPL', 8: 'I-PLC', 9: 'O', 10: 'I-FNB'}
    NUM_LABELS = 11
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# Pos Tag Idn + Prosa
#####
class PosTagIdnDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'B-PR': 0, 'B-CD': 1, 'I-PR': 2, 'B-SYM': 3, 'B-JJ': 4, 'B-DT': 5, 'I-UH': 6, 'I-NND': 7, 'B-SC': 8, 'I-WH': 9, 'I-IN': 10, 'I-NNP': 11, 'I-VB': 12, 'B-IN': 13, 'B-NND': 14, 'I-CD': 15, 'I-JJ': 16, 'I-X': 17, 'B-OD': 18, 'B-RP': 19, 'B-RB': 20, 'B-NNP': 21, 'I-RB': 22, 'I-Z': 23, 'B-CC': 24, 'B-NEG': 25, 'B-VB': 26, 'B-NN': 27, 'B-MD': 28, 'B-UH': 29, 'I-NN': 30, 'B-PRP': 31, 'I-SC': 32, 'B-Z': 33, 'I-PRP': 34, 'I-OD': 35, 'I-SYM': 36, 'B-WH': 37, 'B-FW': 38, 'I-CC': 39, 'B-X': 40}
    INDEX2LABEL = {0: 'B-PR', 1: 'B-CD', 2: 'I-PR', 3: 'B-SYM', 4: 'B-JJ', 5: 'B-DT', 6: 'I-UH', 7: 'I-NND', 8: 'B-SC', 9: 'I-WH', 10: 'I-IN', 11: 'I-NNP', 12: 'I-VB', 13: 'B-IN', 14: 'B-NND', 15: 'I-CD', 16: 'I-JJ', 17: 'I-X', 18: 'B-OD', 19: 'B-RP', 20: 'B-RB', 21: 'B-NNP', 22: 'I-RB', 23: 'I-Z', 24: 'B-CC', 25: 'B-NEG', 26: 'B-VB', 27: 'B-NN', 28: 'B-MD', 29: 'B-UH', 30: 'I-NN', 31: 'B-PRP', 32: 'I-SC', 33: 'B-Z', 34: 'I-PRP', 35: 'I-OD', 36: 'I-SYM', 37: 'B-WH', 38: 'B-FW', 39: 'I-CC', 40: 'B-X'}
    NUM_LABELS = 41
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
    
class PosTagProsaDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'B-PPO': 0, 'B-KUA': 1, 'B-ADV': 2, 'B-PRN': 3, 'B-VBI': 4, 'B-PAR': 5, 'B-VBP': 6, 'B-NNP': 7, 'B-UNS': 8, 'B-VBT': 9, 'B-VBL': 10, 'B-NNO': 11, 'B-ADJ': 12, 'B-PRR': 13, 'B-PRK': 14, 'B-CCN': 15, 'B-$$$': 16, 'B-ADK': 17, 'B-ART': 18, 'B-CSN': 19, 'B-NUM': 20, 'B-SYM': 21, 'B-INT': 22, 'B-NEG': 23, 'B-PRI': 24, 'B-VBE': 25}
    INDEX2LABEL = {0: 'B-PPO', 1: 'B-KUA', 2: 'B-ADV', 3: 'B-PRN', 4: 'B-VBI', 5: 'B-PAR', 6: 'B-VBP', 7: 'B-NNP', 8: 'B-UNS', 9: 'B-VBT', 10: 'B-VBL', 11: 'B-NNO', 12: 'B-ADJ', 13: 'B-PRR', 14: 'B-PRK', 15: 'B-CCN', 16: 'B-$$$', 17: 'B-ADK', 18: 'B-ART', 19: 'B-CSN', 20: 'B-NUM', 21: 'B-SYM', 22: 'B-INT', 23: 'B-NEG', 24: 'B-PRI', 25: 'B-VBE'}
    NUM_LABELS = 26
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)

class PosTagDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(PosTagDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

#####
# Emotion Twitter
#####
class EmotionDetectionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
    INDEX2LABEL = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset

    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        tweet, label = self.data.loc[index,'tweet'], self.data.loc[index,'label']        
        subwords = self.tokenizer.encode(tweet, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(label), tweet
    
    def __len__(self):
        return len(self.data)
        
class EmotionDetectionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(EmotionDetectionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        label_batch = np.full((batch_size, 1), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            label_batch[i] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, label_batch, seq_list

#####
# Entailment UI
#####
class EntailmentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'NotEntail': 0, 'Entail_or_Paraphrase': 1}
    INDEX2LABEL = {0: 'NotEntail', 1: 'Entail_or_Paraphrase'}
    NUM_LABELS = 2
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        df['label'] = df['label'].apply(lambda label: self.LABEL2INDEX[label])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sent_A, sent_B, label = data['sent_A'], data['sent_B'], data['label']
        
        encoded_inputs = self.tokenizer.encode_plus(sent_A, sent_B, add_special_tokens=not self.no_special_token, return_token_type_ids=True)
        subwords, token_type_ids = encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]
        
        return np.array(subwords), np.array(token_type_ids), np.array(label), data['sent_A'] + "|" + data['sent_B']
    
    def __len__(self):
        return len(self.data)
    
        
class EntailmentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(EntailmentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        label_batch = np.zeros((batch_size, 1), dtype=np.int64)

        seq_list = []
        
        for i, (subwords, token_type_ids, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            token_type_batch[i,:len(subwords)] = token_type_ids
            label_batch[i,0] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, token_type_batch, label_batch, seq_list

#####
# Document Sentiment Prosa
#####
class DocumentSentimentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
    INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}
    NUM_LABELS = 3
    
    def load_dataset(self, path): 
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['text','sentiment']
        df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, sentiment = data['text'], data['sentiment']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), data['text']
    
    def __len__(self):
        return len(self.data)    
        
class DocumentSentimentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, sentiment, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            sentiment_batch[i,0] = sentiment
            
            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, sentiment_batch, seq_list

#####
# Keyword Extraction Prosa
#####
class KeywordExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O':0, 'B':1, 'I':2}
    INDEX2LABEL = {0:'O', 1:'B', 2:'I'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class KeywordExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(KeywordExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []

        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# QA Factoid ITB
#####
class QAFactoidDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O':0, 'B':1, 'I':2}
    INDEX2LABEL = {0:'O', 1:'B', 2:'I'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Read file
        dataset = pd.read_csv(path)
        
        # Question and passage are a list of words and seq_label is list of B/I/O
        dataset['question'] = dataset['question'].apply(lambda x: eval(x))
        dataset['passage'] = dataset['passage'].apply(lambda x: eval(x))
        dataset['seq_label'] = dataset['seq_label'].apply(lambda x: [self.LABEL2INDEX[l] for l in eval(x)])

        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        question, passage, seq_label = data['question'],  data['passage'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        token_type_ids = [0]
        
        # Add subwords for question
        for word_idx, word in enumerate(question):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [-1 for i in range(len(subword_list))]
            token_type_ids += [0 for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add intermediate SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        token_type_ids += [0]
        
        # Add subwords
        for word_idx, word in enumerate(passage):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            token_type_ids += [1 for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        token_type_ids += [1]
        
        return np.array(subwords), np.array(token_type_ids), np.array(subword_to_word_indices), np.array(seq_label), ' '.join(question) + "|" + ' '.join(passage)
    
    def __len__(self):
        return len(self.data)
        
class QAFactoidDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(QAFactoidDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[3]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, token_type_ids, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            token_type_batch[i,:len(subwords)] = token_type_ids
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

#####
# ABSA Airy + Prosa
#####
class AspectBasedSentimentAnalysisAiryDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['ac', 'air_panas', 'bau', 'general', 'kebersihan', 'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
    LABEL2INDEX = {'neg': 0, 'neut': 1, 'pos': 2, 'neg_pos': 3}
    INDEX2LABEL = {0: 'neg', 1: 'neut', 2: 'pos', 3: 'neg_pos'}
    NUM_LABELS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    NUM_ASPECTS = 10
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        for aspect in self.ASPECT_DOMAIN:
            df[aspect] = df[aspect].apply(lambda sen: self.LABEL2INDEX[sen])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sentence, labels = data['review'], [data[aspect] for aspect in self.ASPECT_DOMAIN]
        subwords = self.tokenizer.encode(sentence, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['review']
    
    def __len__(self):
        return len(self.data)
    
class AspectBasedSentimentAnalysisProsaDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['fuel', 'machine', 'others', 'part', 'price', 'service']
    LABEL2INDEX = {'negative': 0, 'neutral': 1, 'positive': 2}
    INDEX2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
    NUM_LABELS = [3, 3, 3, 3, 3, 3]
    NUM_ASPECTS = 6
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        for aspect in self.ASPECT_DOMAIN:
            df[aspect] = df[aspect].apply(lambda sen: self.LABEL2INDEX[sen])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sentence, labels = data['sentence'], [data[aspect] for aspect in self.ASPECT_DOMAIN]
        subwords = self.tokenizer.encode(sentence, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['sentence']
    
    def __len__(self):
        return len(self.data)
    
        
class AspectBasedSentimentAnalysisDataLoader(DataLoader):
    def __init__(self, dataset, max_seq_len=512, *args, **kwargs):
        super(AspectBasedSentimentAnalysisDataLoader, self).__init__(dataset=dataset, *args, **kwargs)
        self.num_aspects = dataset.NUM_ASPECTS
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        label_batch = np.zeros((batch_size, self.num_aspects), dtype=np.int64)

        seq_list = []
        
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            label_batch[i,:] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, label_batch, seq_list

#####
# News Categorization Prosa
#####
class NewsCategorizationDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'permasalahan pada bank besar domestik': 0, 'pertumbuhan ekonomi domestik yang terbatas': 1, 'volatilitas harga komoditas utama dunia': 2, 'frekuensi kenaikan fed fund rate (ffr) yang melebihi ekspektasi': 3, 'perubahan kebijakan dan/atau regulasi pada institusi keuangan': 4, 'isu politik domestik': 5, 'permasalahan pada bank besar international': 6, 'perubahan kebijakan pemerintah yang berkaitan dengan fiskal': 7, 'pertumbuhan ekonomi global yang terbatas': 8, 'kebijakan pemerintah yang bersifat sektoral': 9, 'isu politik dan ekonomi luar negeri': 10, 'kenaikan harga volatile food': 11, 'tidak berisiko': 12, 'pergerakan harga minyak mentah dunia': 13, 'force majeure yang memengaruhi operasional sistem keuangan': 14, 'kenaikan administered price': 15}
    INDEX2LABEL = {0: 'permasalahan pada bank besar domestik', 1: 'pertumbuhan ekonomi domestik yang terbatas', 2: 'volatilitas harga komoditas utama dunia', 3: 'frekuensi kenaikan fed fund rate (ffr) yang melebihi ekspektasi', 4: 'perubahan kebijakan dan/atau regulasi pada institusi keuangan', 5: 'isu politik domestik', 6: 'permasalahan pada bank besar international', 7: 'perubahan kebijakan pemerintah yang berkaitan dengan fiskal', 8: 'pertumbuhan ekonomi global yang terbatas', 9: 'kebijakan pemerintah yang bersifat sektoral', 10: 'isu politik dan ekonomi luar negeri', 11: 'kenaikan harga volatile food', 12: 'tidak berisiko', 13: 'pergerakan harga minyak mentah dunia', 14: 'force majeure yang memengaruhi operasional sistem keuangan', 15: 'kenaikan administered price'}
    NUM_LABELS = 16
    
    def load_dataset(self, path):
        dataset = pd.read_csv(path, sep='\t', header=None)
        dataset.columns = ['text', 'label']
        dataset['label'] = dataset['label'].apply(lambda labels: [self.LABEL2INDEX[label] for label in labels.split(',')])
        return dataset
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, labels = data['text'], data['label']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['text']
    
    def __len__(self):
        return len(self.data)    
        
class NewsCategorizationDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NewsCategorizationDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        # Trimmed input based on specified max_len
        if self.max_seq_len < max_seq_len:
            max_seq_len = self.max_seq_len
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        labels_batch = np.zeros((batch_size, NUM_LABELS), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, labels) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            for label in labels:
                labels_batch[i,label] = 1

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, labels_batch, seq_list

#####
# Generation Model Dataset
#####

##
# Machine Translation
##
class MachineTranslationDataset(Dataset):
    # JSON Format
    # [{
    #    'id': 'id_string',
    #    'text': 'input_string',
    #    'label': 'target_string'
    # }, ... ]
    def load_dataset(self, path): 
        data = json.load(open(path, 'r'))
        return data

    def __init__(self, dataset_path, tokenizer, swap_source_target, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.swap_source_target = swap_source_target
    
    def __getitem__(self, index):
        data = self.data[index]
        id, text, label = data['id'], data['text'], data['label']
        input_subwords = self.tokenizer.encode(text.lower(), add_special_tokens=False)
        label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
        if self.swap_source_target:
            return data['id'], label_subwords, input_subwords
        else:
            return data['id'], input_subwords, label_subwords
    
    def __len__(self):
        return len(self.data)

##
# Summarization
##
class SummarizationDataset(Dataset):
    # JSON Format
    # [{
    #    'id': 'id_string',
    #    'text': 'input_string',
    #    'label': 'target_string'
    # }, ... ]
    # def load_dataset(self, path): 
    #     data = []
    #     with open(path, "r", encoding="utf-8") as f:
    #         for line in f:
    #             arr = line.replace("\n", "").split("\t")
    #             id, text, label = arr
    #             data.append({"id": id, "text": text, "label": label})
    #     return data
    
    # def __init__(self, dataset_path, tokenizer, *args, **kwargs):
    #     self.data = self.load_dataset(dataset_path)
    #     self.tokenizer = tokenizer
    
    # def __getitem__(self, index):
    #     data = self.data[index]
    #     id, text, label = data['id'], data['text'], data['label']
    #     input_subwords = self.tokenizer.encode(text.lower(), add_special_tokens=False)
    #     label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
    #     return data['id'], np.array(input_subwords), np.array(label_subwords)
    
    # def __len__(self):
    #     return len(self.data)
    def load_dataset(self, path): 
        data = json.load(open(path, 'r'))
        return data
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        data = self.data[index]
        id, text, label = data['id'], data['text'], data['label']
        input_subwords = self.tokenizer.encode(text.lower(), add_special_tokens=False)
        label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
        return data['id'], np.array(input_subwords), np.array(label_subwords)
    
    def __len__(self):
        return len(self.data)

##
# Chit Chat
##
class ChitChatDataset(Dataset):
    # JSON Format
    # [{
    #    'persona': ['Saya suka merombak rumah.',...,...,'liburan favorit saya adalah halloween.'],
    #    'dialogue': [['Hai apa ka ... hobi favorit saya.'],[...],...,[...]]
    # }, ... ]
    def __init__(self, dataset_path, tokenizer, speaker_1_id=5, speaker_2_id=6, max_token_length = 512, *args, **kwargs):
        self.data = self.load_dataset(dataset_path, tokenizer, speaker_1_id, speaker_2_id, max_token_length)
        self.max_len = max(len(x["tokenized_history"]) for x in self.data)  
        self.tokenizer = tokenizer

    def load_dataset(self, path, tokenizer, speaker_1_id, speaker_2_id, max_token_length): 
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj.lower()))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for i in range(len(raw_data)):
            raw_data[i].pop('persona')

        # structuring the result data
        i = 0
        data = []
        for dial in raw_data:
            for turn in dial["dialogue"]:
                data.append({'id':i})
                i+=1

        # split raw data for history and response
        i = 0
        for dial in raw_data:
            dialogue_history = []
            for turn in dial["dialogue"]:
                dialogue_history.append(turn[0])
                data[i]['text_history'] = dialogue_history.copy()
                data[i]['text_response'] = turn[1]
                dialogue_history.append(turn[1])
                i+=1

        # tokenize
        dataset_tokenized = tokenize(raw_data)

        # split tokenized data for history and response
        i = 0
        for dial in dataset_tokenized:
            dialogue_history = []
            for turn in dial["dialogue"]:
                dialogue_history.append(turn[0])
                data[i]['tokenized_history'] = dialogue_history.copy()
                data[i]['tokenized_response'] = turn[1]
                dialogue_history.append(turn[1])
                i+=1

        # limit the chit chat history according to the max_seq_len
        for i in range(len(data)):
            token_length = sum([len(turn) for turn in data[i]['tokenized_history']])
            for remove_i_first_turn in range(len(data[i]['tokenized_history'])):
                if token_length > max_token_length - 2 - 2*len(data[i]['tokenized_history']):
                    data[i]['tokenized_history'] = data[i]['tokenized_history'][(remove_i_first_turn+1):]
                    data[i]['text_history'] = data[i]['text_history'][(remove_i_first_turn+1):]
                    token_length = sum([len(turn) for turn in data[i]['tokenized_history']])
                else:
                    break

        # speaker_token_assignment
        for i in range(len(data)):
            instance_text = self.speaker_token_assignment(data[i]['text_history'], \
                                                          data[i]['text_response'], \
                                                          speaker_1_id, speaker_2_id, tokenizer, tokenized=False)
            instance_tokenized = self.speaker_token_assignment(data[i]['tokenized_history'], \
                                                               data[i]['tokenized_response'], \
                                                               speaker_1_id, speaker_2_id, tokenizer, tokenized=True)
            data[i]['text_history'] = instance_text["history"]
            data[i]['text_response'] = instance_text["response"]
            data[i]['tokenized_history'] = instance_tokenized["history"]
            data[i]['tokenized_response'] = instance_tokenized["response"]

        return data

    def speaker_token_assignment(self, history, reply, speaker1, speaker2, tokenizer, tokenized=False):
        SPECIAL_TOKENS = [speaker1, speaker2]

        if tokenized:
            sequence = [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(history)]
            response = [reply]
        else:
            sequence = [[speaker2 if i % 2 else speaker1] + [s] for i, s in enumerate(history)]
            response = [[reply]]

        instance = {}
        instance["history"] = list(chain(*sequence)) if len(sequence) > 0 else sequence
        instance["response"] = list(chain(*response)) if len(response) > 0 else response

        return instance
    
    def __getitem__(self, index):
        data = self.data[index]
        id, text, label = data['id'], data['tokenized_history'], data['tokenized_response']
        return data['id'], list(text), list(label)
    
    def __len__(self):
        return len(self.data)

##
# Question Answering
##
class QuestionAnsweringDataset(Dataset):
    # JSON Format
    # [{
    #    'id': 'id_string',
    #    'passage': 'input_string',
    #    'question': 'input_string',
    #    'label': 'target_string'
    # }, ... ]
    def load_dataset(self, path): 
        data = json.load(open(path, 'r'))
        return data
    
    def __init__(self, dataset_path, tokenizer, separator_id, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.separator_id = separator_id
    
    def __getitem__(self, index):
        data = self.data[index]
        id, context, question, label = data['id'], data['context'], data['question'], data['label']
        context_subwords = self.tokenizer.encode(context.lower(), add_special_tokens=False)
        question_subwords = self.tokenizer.encode(question.lower(), add_special_tokens=False)
        
        input_subwords = np.concatenate([context_subwords, [self.separator_id], question_subwords]).tolist()
        label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
        return data['id'], input_subwords, label_subwords
    
    def __len__(self):
        return len(self.data)
    
###
# Generation Data Loader
###
class GenerationDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, src_lid_token_id=1, tgt_lid_token_id=2, label_pad_token_id=-100, model_type='indo-bart', tokenizer=None, *args, **kwargs):
        super(GenerationDataLoader, self).__init__(*args, **kwargs)
    
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.src_lid_token_id = src_lid_token_id
        self.tgt_lid_token_id = tgt_lid_token_id
        self.label_pad_token_id = label_pad_token_id
               
        if model_type == 'transformer':
            self.collate_fn = self._bart_collate_fn
        elif model_type == 'indo-bart':
            self.collate_fn = self._bart_collate_fn
        elif model_type == 'indo-t5':
            self.T5_TOKEN_ID_TO_LANG_MAP = {
                tokenizer.indonesian_token_id: 'indonesian',
                tokenizer.english_token_id: 'english',
                tokenizer.sundanese_token_id: 'sundanese',
                tokenizer.javanese_token_id: 'javanese'
            }
            
            if self.tokenizer is not None:
                src_lang, tgt_lang = self.T5_TOKEN_ID_TO_LANG_MAP[src_lid_token_id], self.T5_TOKEN_ID_TO_LANG_MAP[tgt_lid_token_id]
                self.t5_prefix =np.array(self.tokenizer.encode(f'translate {src_lang} to {tgt_lang}: ', add_special_tokens=False))
            
            self.collate_fn = self._t5_collate_fn
        elif model_type == 'indo-gpt2':
            self.collate_fn = self._gpt2_collate_fn
        elif model_type == 'baseline-mbart':
            self.collate_fn = self._baseline_mbart_collate_fn
        elif model_type == 'baseline-mt5':
            self.collate_fn = self._baseline_mt5_collate_fn
        else:
            raise ValueError(f'Unknown model_type `{model_type}`')
            
    def _bart_collate_fn(self, batch):
        ####
        # We make a slight format error during the pre-training, our pretrain decoder format is '<langid><bos><sent><eos>',
        #   but to ensure we have same number of prediction subword limit with mBART model (especially for summarization) 
        #   and also compatibility with other library, we then decide the resulting format will be same as mBART standard:
        # encoder input
        # <sent><eos><langid>
        # decoder input - 
        # <langid><sent><eos>
        # decoder output
        # <sent><eos><langid>
        ###
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch)) + 2) # +2 for eos, and langid
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 2) # +2 for eos, and langid
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len-2], label_seq[:max_dec_len - 2]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + 2] = 1
            dec_mask_batch[i,:len(label_seq) + 2] = 1
            
            # Assign special token to encoder input
            enc_batch[i,len(input_seq)] = self.eos_token_id
            enc_batch[i,1+len(input_seq)] = self.src_lid_token_id
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.tgt_lid_token_id
            dec_batch[i,1+len(label_seq)] = self.eos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            label_batch[i,1+len(label_seq)] = self.tgt_lid_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch

    def _t5_collate_fn(self, batch):
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch))  + len(self.t5_prefix))
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 1)
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len - len(self.t5_prefix)], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,len(self.t5_prefix):len(self.t5_prefix) + len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + len(self.t5_prefix)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to encoder input
            enc_batch[i,:len(self.t5_prefix)] = self.t5_prefix
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch
#         return id_batch, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch

    def _gpt2_collate_fn(self, batch): 
        ####
        # GPT2 decoder only format:
        # Training  : <src_sent><bos><tgt_sent><eos>
        # Inference : <src_sent><bos>
        #
        # Training sequence & mask are stored in dec_batch and dec_mask_batch respectively
        # Inference sequence & mask are stored in enc_batch and enc_mask_batch respectively
        ###
        batch_size = len(batch)
        max_enc_len = np.int32(min(self.max_seq_len, max(map(lambda x: len(x[1]), batch)) + 1))
        max_dec_len = np.int32(min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 1))
        max_len = max_enc_len + max_dec_len
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_len), self.pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_len), 0, dtype=np.float32)
        label_batch = np.full((batch_size, max_len), self.label_pad_token_id, dtype=np.int64) 
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
#             if max_len == self.max_seq_len:
#                 input_seq = input_seq[:max_len-len(label_seq)]
            
#             # Assign content & special token to encoder batch (for inference)
#             enc_batch[i,0] = self.bos_token_id
#             enc_batch[i,1:1 + len(input_seq)] = input_seq
#             enc_batch[i,1 + len(input_seq)] = self.eos_token_id
#             enc_batch[i,1 + len(input_seq) + 1] = self.bos_token_id
                                                
#             # Assign content & special token to decoder batch (for training)
#             # dec_batch[i,0] = self.src_lid_token_id
#             dec_batch[i,0] = self.bos_token_id
#             dec_batch[i,1:1 + len(input_seq)] = input_seq
#             dec_batch[i,1 + len(input_seq)] = self.eos_token_id
#             # dec_batch[i,1 + len(input_seq) + 1] = self.tgt_lid_token_id
#             dec_batch[i,1 + len(input_seq) + 1] = self.bos_token_id
#             dec_batch[i,1 + len(input_seq) + 2: 1 + len(input_seq) + 2 + len(label_seq)] = label_seq
                                                
#             # Assign Mask for encoder & decoder batch
#             enc_mask_batch[i,:1 + len(input_seq) + 2] = 1
#             dec_mask_batch[i,:1 + len(input_seq) + 2 + len(label_seq)] = 1
            
#             # Assign content & special token to label batch, ignore the input prefix until <tgt_lang_id>
#             label_batch[i,1 + len(input_seq) + 1:1 + len(input_seq) + 1 + len(label_seq)] = label_seq
#             label_batch[i,1 + len(input_seq) + 1 + len(label_seq)] = self.eos_token_id

            input_seq = input_seq[:self.max_seq_len - 1]
            label_seq = label_seq[:self.max_seq_len - 1]

            # Assign content & special token to encoder batch (for inference)
            enc_batch[i,:len(input_seq)] = input_seq
            enc_batch[i,max_enc_len - 1] = self.bos_token_id
                                                
            # Assign content & special token to decoder batch (for training)
            # dec_batch[i,0] = self.src_lid_token_id
            dec_batch[i,:len(input_seq)] = input_seq
            dec_batch[i,max_enc_len-1] = self.bos_token_id
            # dec_batch[i,1 + len(input_seq) + 1] = self.tgt_lid_token_id
            dec_batch[i,max_enc_len:max_enc_len+len(label_seq)] = label_seq
                                                
            # Assign Mask for encoder batch
            enc_mask_batch[i,:len(input_seq)] = 1
            enc_mask_batch[i,max_enc_len - 1] = 1
            
            # Assign Mask for decoder batch
            dec_mask_batch[i,:len(input_seq)] = 1
            dec_mask_batch[i,max_enc_len - 1] = 1
            dec_mask_batch[i,max_enc_len:max_enc_len+len(label_seq)] = 1
            
            # Assign content & special token to label batch, no need to shift left as it will be done inside the GPT2LMHeadModel
            label_batch[i,max_enc_len:max_enc_len+len(label_seq)] = label_seq
            label_batch[i,max_enc_len+len(label_seq)] = self.eos_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch

    def _baseline_mbart_collate_fn(self, batch):
        ####
        # We follow mBART pre-training format, there is a discussions for the mBART tokenizer (https://github.com/huggingface/transformers/issues/7416)
        #   which mentioned the format of the labels should be: <langid><sent><eos><langid>
        #   and the mBART model will add the <langid> as a prefix to create the decoder_input_ids during the forward function.
        # 
        # In order to make it consistent and easier to understand with the other models, we keep our dataloader similar to our IndoNLG models
        #   with the following output format:
        # encoder input
        # <sent><eos><langid>
        # decoder input
        # <langid><sent><eos>
        # decoder output
        # <sent><eos><langid>
        ###
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch)) + 2) # + 2 for eos and langid
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 2) # + 2 for eos and langid
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len-2], label_seq[:max_dec_len - 2]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,0:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + 2] = 1
            dec_mask_batch[i,:len(label_seq) + 2] = 1
            
            # Assign special token to encoder input
            enc_batch[i,len(input_seq)] = self.eos_token_id
            enc_batch[i,1+len(input_seq)] = self.src_lid_token_id
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.tgt_lid_token_id
            dec_batch[i,1+len(label_seq)] = self.eos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            label_batch[i,1+len(label_seq)] = self.tgt_lid_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch
#         return id_batch, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch
        
    def _baseline_mt5_collate_fn(self, batch):
        ####
        # As mT5 is only trained on MLM without additional language identifier, we can actually fine tune without prefix
        # In this case we make the input output format as follow
        # encoder input
        # <sent>
        # decoder input
        # <bos><sent>
        # decoder output
        # <sent><eos>
        ###
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch))) # No additional token is needed
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 1) # + 1 for bos / eos
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,0:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch
