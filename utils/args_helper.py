from utils.data_utils import AspectExtractionDataset, AspectExtractionDataLoader
from utils.data_utils import NerGritDataset, NerProsaDataset, NerDataLoader
from utils.data_utils import PosTagIdnDataset, PosTagProsaDataset, PosTagDataLoader
from utils.data_utils import EmotionDetectionDataset, EmotionDetectionDataLoader
from utils.data_utils import EntailmentDataset, EntailmentDataLoader
from utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
from utils.data_utils import KeywordExtractionDataset, KeywordExtractionDataLoader
from utils.data_utils import NewsCategorizationDataset, NewsCategorizationDataLoader
from utils.data_utils import QAFactoidDataset, QAFactoidDataLoader
from utils.data_utils import AspectBasedSentimentAnalysisAiryDataset, AspectBasedSentimentAnalysisProsaDataset, AspectBasedSentimentAnalysisDataLoader
from utils.data_utils import MachineTranslationDataset, SummarizationDataset, ChitChatDataset, QuestionAnsweringDataset, GenerationDataLoader

from utils.metrics import emotion_detection_metrics_fn, aspect_extraction_metrics_fn, ner_metrics_fn, pos_tag_metrics_fn, entailment_metrics_fn, document_sentiment_metrics_fn, keyword_extraction_metrics_fn, news_categorization_metrics_fn, qa_factoid_metrics_fn, absa_metrics_fn, generation_metrics_fn, tydi_qa_metrics_fn
from utils.forward_fn import forward_sequence_classification, forward_word_classification, forward_sequence_multi_classification, forward_generation

from nltk.tokenize import TweetTokenizer
from argparse import ArgumentParser

###
# args functions
###
def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.keys():
        if opts[key]:
            print('{:>30}: {:<50}'.format(key, str(opts[key])).center(80))
    print('=' * 80)
    
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--model_dir", type=str, default="save/", help="Model directory")
    parser.add_argument("--dataset", type=str, default='emotion-twitter', help="Choose between emotion-twitter, absa-airy, term-extraction-airy, ner-grit, pos-idn, entailment-ui, doc-sentiment-prosa, keyword-extraction-prosa, qa-factoid-itb, news-category-prosa, ner-prosa, pos-prosa")
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_type", type=str, default=None, help="Type of the model (`transformer`, `indo-bart`, `indo-t5`, `indo-gpt2`, `baseline-mbart`, or `baseline-mt5`)")
    parser.add_argument("--grad_accumulate", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path, url or short name of the model")
    parser.add_argument("--beam_size", type=int, default=5, help="Size of beam search")
    parser.add_argument("--max_history", type=int, default=1000000000, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max number of tokens")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for testing")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--vocab_path", type=str, default='./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model', help="Vocab path")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=10.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default='cuda0', help="Device (cuda or cpu)")
    parser.add_argument("--fp16", default=False, action='store_true', help="use FP16 to reduce computational and memory costs")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--weight_tie", action='store_true', help="Use weight tie")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--early_stop", type=int, default=3, help="Step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--force", action='store_true', help="force to rewrite experiment folder")
    parser.add_argument("--no_special_token", action='store_true', help="not adding special token as the input")
    parser.add_argument("--lower", action='store_true', help="lower case")
    
    parser.add_argument("--freeze_encoder", default=False, action='store_true', help="whether to freeze encoder or decoder")
    parser.add_argument("--freeze_decoder", default=False, action='store_true', help="whether to freeze encoder or decoder")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")

    args = vars(parser.parse_args())
    print_opts(args)
    return args

def get_eval_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--model_dir", type=str, default="./save", help="Model directory")
    parser.add_argument("--dataset", type=str, default='emotion-twitter', help="Choose between emotion-twitter, absa-airy, term-extraction-airy, ner-grit, pos-idn, entailment-ui, doc-sentiment-prosa, keyword-extraction-prosa, qa-factoid-itb, news-category-prosa, ner-prosa, pos-prosa")
    parser.add_argument("--model_type", type=str, default="bert-base-multilingual-uncased", help="Type of the model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max number of tokens")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--no_special_token", action='store_true', help="not adding special token as the input")
    parser.add_argument("--lower", action='store_true', help="lower case")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default='cuda', help="Device (cuda or cpu)")

    args = vars(parser.parse_args())
    print_opts(args)
    return args

#TODO: Need to change it into a json or something else that are easily extendable
def append_model_args(args):
    if args['model_type'] == 'indo-bart':
        args['separator_id'] = 4 # <0x00>
        args['speaker_1_id'] = 5 # <0x01>
        args['speaker_2_id'] = 6 # <0x02>
    elif args['model_type'] == 'indo-gpt2':
        args['separator_id'] = 4 # <0x00>
        args['speaker_1_id'] = 5 # <0x01>
        args['speaker_2_id'] = 6 # <0x02>
    elif args['model_type'] == 'baseline-mbart':
        args['separator_id'] = 2 # </s> following mBart pretraining
        args['speaker_1_id'] = 250055 # Additional token <speaker_1> 
        args['speaker_2_id'] = 250056 # Additional token <speaker_2>
    elif args['model_type'] == 'baseline-mt5':
        args['separator_id'] = 3 # <0x00> | Extra token <extra_token_2> 250097
        args['speaker_1_id'] = 4 # <0x01> | Extra token <extra_token_1> 250098
        args['speaker_2_id'] = 5 # <0x02> | Extra token <extra_token_0> 250099
    elif args['model_type'] == 'transformer':
        args['separator_id'] = 4 # <0x00>
        args['speaker_1_id'] = 5 # <0x01>
        args['speaker_2_id'] = 6 # <0x02>
    else: # if args['model_type'] == 'bart':
        raise ValueError('Unknown model type')
    return args

def append_dataset_args(args):
    if args['dataset'] == "mt-javnrf-inzntv":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_JAVNRF_INZNTV/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_JAVNRF_INZNTV/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_JAVNRF_INZNTV/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[java]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "jv_JV"
        args['target_lang_bart'] = "id_ID"
        args['swap_source_target'] = False
        args['k_fold'] = 1
    elif args['dataset'] == "mt-sunibs-inzntv":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_SUNIBS_INZNTV/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_SUNIBS_INZNTV/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_SUNIBS_INZNTV/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[sunda]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "su_SU"
        args['target_lang_bart'] = "id_ID"
        args['swap_source_target'] = False
        args['k_fold'] = 1
    elif args['dataset'] == "mt-enkjv-inzntv":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_ENGKJV_INZNTV/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_ENGKJV_INZNTV/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_ENGKJV_INZNTV/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[english]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "en_XX"
        args['target_lang_bart'] = "id_ID"
        args['swap_source_target'] = False
        args['k_fold'] = 1
    elif args['dataset'] == "mt-ted-multi":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_TED_MULTI/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_TED_MULTI/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_TED_MULTI/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[english]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "en_XX"
        args['target_lang_bart'] = "id_ID"
        args['swap_source_target'] = False
        args['k_fold'] = 1
    elif args['dataset'] == "mt-imd-news":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_IMD_NEWS/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_IMD_NEWS/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_IMD_NEWS/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[english]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "en_XX"
        args['target_lang_bart'] = "id_ID"
        args['swap_source_target'] = False
        args['k_fold'] = 1
    elif args['dataset'] == "mt-imd-religion":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_IMD_RELIGION/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_IMD_RELIGION/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_IMD_RELIGION/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[english]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "en_XX"
        args['target_lang_bart'] = "id_ID"
        args['swap_source_target'] = False
        args['k_fold'] = 1
    elif args['dataset'] == "mt-javnrf-inzntv-swap":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_JAVNRF_INZNTV/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_JAVNRF_INZNTV/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_JAVNRF_INZNTV/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[java]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "jv_JV"
        args['swap_source_target'] = True
        args['k_fold'] = 1
    elif args['dataset'] == "mt-sunibs-inzntv-swap":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_SUNIBS_INZNTV/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_SUNIBS_INZNTV/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_SUNIBS_INZNTV/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[sunda]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "su_SU"
        args['swap_source_target'] = True
        args['k_fold'] = 1
    elif args['dataset'] == "mt-enkjv-inzntv-swap":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_ENGKJV_INZNTV/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_ENGKJV_INZNTV/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_ENGKJV_INZNTV/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[english]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "en_XX"
        args['swap_source_target'] = True
        args['k_fold'] = 1
    elif args['dataset'] == "mt-ted-multi-swap":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_TED_MULTI/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_TED_MULTI/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_TED_MULTI/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[english]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "en_XX"
        args['swap_source_target'] = True
        args['k_fold'] = 1
    elif args['dataset'] == "mt-imd-news-swap":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_IMD_NEWS/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_IMD_NEWS/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_IMD_NEWS/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[english]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "en_XX"
        args['swap_source_target'] = True
        args['k_fold'] = 1
    elif args['dataset'] == "mt-imd-religion-swap":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/MT_IMD_RELIGION/train_preprocess.json'
        args['valid_set_path'] = './dataset/MT_IMD_RELIGION/valid_preprocess.json'
        args['test_set_path'] = './dataset/MT_IMD_RELIGION/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[english]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "en_XX"
        args['swap_source_target'] = True
        args['k_fold'] = 1
    elif args['dataset'] == "indosum":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/indosum/train.json'
        args['valid_set_path'] = './dataset/indosum/dev.json'
        args['test_set_path'] = './dataset/indosum/test.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "id_ID"
        args['k_fold'] = 1
    elif args['dataset'] == "xpersona":
        args['dataset_class'] = ChitChatDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'SacreBLEU'
        args['train_set_path'] = './dataset/xpersona/Id_persona_train_corrected.json'
        args['valid_set_path'] = './dataset/xpersona/Id_persona_split_valid_human_annotated.json'
        args['test_set_path'] = './dataset/xpersona/Id_persona_split_test_human_annotated.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "id_ID"
        args['k_fold'] = 1
    elif args['dataset'] == "liputan6_canonical":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'BLEU'
        args['train_set_path'] = './dataset/liputan6/canonical_train.json'
        args['valid_set_path'] = './dataset/liputan6/canonical_dev.json'
        args['test_set_path'] = './dataset/liputan6/canonical_test.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "id_ID"
        args['k_fold'] = 1
    elif args['dataset'] == "liputan6_xtreme":
        args['dataset_class'] = MachineTranslationDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = generation_metrics_fn
        args['valid_criterion'] = 'BLEU'
        args['train_set_path'] = './dataset/liputan6/canonical_train.json'
        args['valid_set_path'] = './dataset/liputan6/xtreme_dev.json'
        args['test_set_path'] = './dataset/liputan6/xtreme_test.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "id_ID"
        args['k_fold'] = 1
    elif args['dataset'] == "tydi-qa":
        args['dataset_class'] = QuestionAnsweringDataset
        args['dataloader_class'] = GenerationDataLoader
        args['forward_fn'] = forward_generation
        args['metrics_fn'] = tydi_qa_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/question_answering/train_preprocess.json'
        args['valid_set_path'] = './dataset/question_answering/valid_preprocess.json'
        args['test_set_path'] = './dataset/question_answering/test_preprocess.json'
        # args['vocab_path'] = "./vocab/IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
        args['source_lang'] = "[indonesia]"
        args['target_lang'] = "[indonesia]"
        args['source_lang_bart'] = "id_ID"
        args['target_lang_bart'] = "id_ID"
        args['k_fold'] = 1
    else:
        raise ValueError(f'Unknown dataset name `{args["dataset"]}`')
    return args
