import itertools
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from .conlleval import conll_evaluation
import datasets
from nltk import word_tokenize

bleu = datasets.load_metric('bleu')
rouge = datasets.load_metric('rouge')
sacrebleu = datasets.load_metric('sacrebleu')
squad_v2_metric = datasets.load_metric('squad_v2')

def emotion_detection_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def aspect_extraction_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = f1
    metrics["REC"] = rec
    metrics["PRE"] = pre
    return metrics

def ner_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = f1
    metrics["REC"] = rec
    metrics["PRE"] = pre
    return metrics

def pos_tag_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = f1
    metrics["REC"] = rec
    metrics["PRE"] = pre
    return metrics

def entailment_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def document_sentiment_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def keyword_extraction_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = f1
    metrics["REC"] = rec
    metrics["PRE"] = pre
    return metrics

def qa_factoid_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = f1
    metrics["REC"] = rec
    metrics["PRE"] = pre
    return metrics

def absa_metrics_fn(list_hyp, list_label):
    # hyp and label are both list (multi label), flatten the list
    list_hyp = list(itertools.chain.from_iterable(list_hyp))
    list_label = list(itertools.chain.from_iterable(list_label))
    
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def news_categorization_metrics_fn(list_hyp, list_label):
    # hyp and label are both list (multi label), flatten the list
    list_hyp = list(itertools.chain.from_iterable(list_hyp))
    list_label = list(itertools.chain.from_iterable(list_label))
    
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics

def generation_metrics_fn(list_hyp, list_label):
    # hyp and label are both list of string
    list_hyp_bleu = list(map(lambda x: word_tokenize(x), list_hyp))
    list_label_bleu = list(map(lambda x: [word_tokenize(x)], list_label))
    list_label_sacrebleu = list(map(lambda x: [x], list_label))
    
    metrics = {}
    metrics["BLEU"] = bleu._compute(list_hyp_bleu, list_label_bleu)['bleu'] * 100
    metrics["SacreBLEU"] = sacrebleu._compute(list_hyp, list_label_sacrebleu)['score']
    
    rouge_score = rouge._compute(list_hyp,list_label)
    metrics["ROUGE1"] = rouge_score['rouge1'].mid.fmeasure * 100
    metrics["ROUGE2"] = rouge_score['rouge2'].mid.fmeasure * 100
    metrics["ROUGEL"] = rouge_score['rougeL'].mid.fmeasure * 100
    metrics["ROUGELsum"] = rouge_score['rougeLsum'].mid.fmeasure * 100
    return metrics

def tydi_qa_metrics_fn(list_hyp, list_label):
    qa_hyps = [{'prediction_text': hyp, 'id': str(id), 'no_answer_probability':0} for id, hyp in enumerate(list_hyp)]
    qa_labels = [{'answers': {'answer_start': [0], 'text': [label]}, 'id': str(id)} for id, label in enumerate(list_label)]
    squad_v2_score = squad_v2_metric.compute(predictions=qa_hyps, references=qa_labels)
    
    metrics = {}
    metrics["EM"] = squad_v2_score['exact']
    metrics["F1"] = squad_v2_score['f1']
    return metrics