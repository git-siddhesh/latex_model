
import sys
import pandas as pd
from pathlib import Path
from datasets import Dataset 
from prettytable import PrettyTable
from transformers import (
    AutoTokenizer, 
    DebertaV2Tokenizer,
    MistralConfig,
    MistralForCausalLM,
    LlamaTokenizerFast,
    DataCollatorForLanguageModeling,
)

import torch
import os
import time

torch.manual_seed(191009)
import pickle

from pynvml import *
def gpu_usage(logger):
    def print_gpu_utilization(logger):
        nvmlInit()
        logger.info(f"GPU memory occupied from nvmlInit: {nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).used//1024**2} MB.")
    print("+----------------------------------------------------------------------------------+")
    a,b = torch.cuda.mem_get_info()
    gpu_mem_usage = (b-a)/(2**20)
    logger.info(f"GPU memory usage before cleaning cache: {gpu_mem_usage:.2f} MB")
    torch.cuda.empty_cache()
    a,b = torch.cuda.mem_get_info()
    gpu_mem_usage = (b-a)/(2**20)
    logger.info(f"GPU memory usage after cleaning cache: {gpu_mem_usage:.2f} MB")
    print_gpu_utilization(logger)
    print("+----------------------------------------------------------------------------------+")


# code_path = "/home/dosisiddhesh/MISTRAL_EXP/mistral-src"
data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/latex.csv"
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/model/mistral-7B-v0.1")  # model and tokenizer location
tokenizer_path = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model" # sentencepiece tokenizer path
# sys.path.append(code_path)  # append the path where mistral-src was cloned
# from mistral.tokenizer import Tokenizer
# from mistral.model import Transformer, ModelArgs
import logging
import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(stream=sys.stdout)  
        self.setLevel(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, end='') 
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class Parameter:
    def __init__(self, name, value, use_cache=True):
        self.name = name
        self.D_emb,self.Vocal,self.d_head,self.d_FF,self.N_Layer,self.N_Head,self.KV_Head,self.Window = value
        self.use_cache = use_cache
    
    # pretty print the parameters in a table
    def print_parameters(self, logger=None):
        table = PrettyTable(["Parameter", "Value"])
        table.add_row(["MODEL EMBEDDING DIM", self.D_emb])
        table.add_row(["VOCABULARY", self.Vocal])
        table.add_row(["PER HEAD DIM", self.d_head])
        table.add_row(["FEED-FORWARD HIDDEN DIM", self.d_FF])
        table.add_row(["NUMBER OF LAYERS", self.N_Layer])
        table.add_row(["NUMBER OF Q-HEAD", self.N_Head])
        table.add_row(["NUMBER OF KV-HEAD", self.KV_Head])
        table.add_row(["WINDOW SIZE", self.Window])
        table.add_row(["use_cache", self.use_cache])
        logger.info(f'\n{table}')

class HyperParams:
    def __init__(self, epoch = 1, learning_rate = 3e-4, model_id = "mistral", **kwargs):
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.model_id = model_id
        self.weight_decay=kwargs.get('weight_decay',0.1)  
        self.warmup_steps=kwargs.get('warmup_steps', 200)
        self.lr_scheduler_type=kwargs.get('lr_scheduler_type', "linear") #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
        self.BATCH_SIZE=kwargs.get('BATCH_SIZE', 2)
        self.tokenizer_batch_size=kwargs.get('tokenizer_batch_size', 2)
        self.eval_steps=kwargs.get('eval_steps', 50) # Adjust as needed1
        self.logging_steps=kwargs.get('logging_steps', 50)  # Adjust as needed
        self.save_steps=kwargs.get('save_steps', 200)
        self.save_total_limit =kwargs.get('save_total_limit', 1)
        self.max_seq_length=kwargs.get('max_seq_length', 1024)
        self.eval_batch_size=kwargs.get('eval_batch_size',1) #2
        self.EVAL_ACCUMULATION_STEPS = kwargs.get('EVAL_ACCUMULATION_STEPS', 1)
        self.eval_frac=kwargs.get('eval_frac', 0.1)
    
    # pretty print the hyperparameters in a table
    def print_hyperparameters(self, logger=None):
        table = PrettyTable(["HyperParameter", "Value"])
        table.add_row(["EPOCHS", self.epochs])
        table.add_row(["LEARNING RATE", self.learning_rate])
        table.add_row(["WEIGHT DECAY", self.weight_decay])
        table.add_row(["WARMUP STEPS", self.warmup_steps])
        table.add_row(["LR SCHEDULER TYPE", self.lr_scheduler_type])
        table.add_row(["TRAIN BATCH SIZE", self.BATCH_SIZE])
        table.add_row(["EVAL BATCH SIZE", self.eval_batch_size])
        table.add_row(["TOKENIZER BATCH SIZE", self.tokenizer_batch_size])
        table.add_row(["EVAL ACCUMULATION STEPS", self.EVAL_ACCUMULATION_STEPS])
        table.add_row(["EVAL STEPS", self.eval_steps])
        table.add_row(["SAVE STEPS", self.save_steps])
        table.add_row(["LOGGING STEPS", self.logging_steps])
        table.add_row(["SAVE TOTAL LIMIT", self.save_total_limit])
        table.add_row(["MAX SEQ LENGTH", self.max_seq_length])
        table.add_row(["EVAL FRAC", self.eval_frac])
        logger.info(f'\n{table}')

    
class MyModel(Parameter, HyperParams):
    def __init__(self, model_id="mistral", hp=None, param=None):
        self.model_id = model_id
        # set the parameters and hyperparameters
        self.param = param
        self.hp = hp
        self.args = None
        if hp is not None:
            self.model_name = f"{self.model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_{hp.lr_scheduler_type}_wt_decay_{hp.weight_decay}_warmup_st_{hp.warmup_steps}"
            # add parameters to the model too in the name
        if param is not None:           
            self.model_name += f"_emb_{param.D_emb}_V_{param.Vocal}_Dhead_{param.d_head}_FF_{param.d_FF}_L_{param.N_Layer}_N_Head_{param.N_Head}_KV_Head_{param.KV_Head}_W_{param.Window}"
    
    # def get_model_name(self,hp):
    #     self.model_name = f"{self.model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_{hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"

    '''vocab_size (int, optional, defaults to 32000) — Vocabulary size of the Mistral model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling MistralModel
    
    hidden_size (int, optional, defaults to 4096) — Dimension of the hidden representations.
    
    intermediate_size (int, optional, defaults to 14336) — Dimension of the MLP representations.
    
    num_hidden_layers (int, optional, defaults to 32) — Number of hidden layers in the Transformer encoder.
    
    num_attention_heads (int, optional, defaults to 32) — Number of attention heads for each attention layer in the Transformer encoder.
    
    num_key_value_heads (int, optional, defaults to 8) — This is the number of key_value heads that should be used to implement Grouped Query Attention. If num_key_value_heads=num_attention_heads, the model will use Multi Head Attention (MHA), if num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to 8`.
    
    hidden_act (str or function, optional, defaults to "silu") — The non-linear activation function (function or string) in the decoder.
    
    max_position_embeddings (int, optional, defaults to 4096*32) — The maximum sequence length that this model might ever be used with. Mistral’s sliding window attention allows sequence of up to 4096*32 tokens.
    initializer_range (float, optional, defaults to 0.02) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    
    rms_norm_eps (float, optional, defaults to 1e-06) — The epsilon used by the rms normalization layers.
    
    use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if config.is_decoder=True.
    
    pad_token_id (int, optional) — The id of the padding token.
    
    bos_token_id (int, optional, defaults to 1) — The id of the “beginning-of-sequence” token.
    
    eos_token_id (int, optional, defaults to 2) — The id of the “end-of-sequence” token.
    
    tie_word_embeddings (bool, optional, defaults to False) — Whether the model’s input and output word embeddings should be tied.
    rope_theta (float, optional, defaults to 10000.0) — The base period of the RoPE embeddings.
    sliding_window (int, optional, defaults to 4096) — Sliding window attention window size. If not specified, will default to 4096.
    attention_dropout (float, optional, defaults to 0.0) — The dropout ratio for the attention probabilities.
    '''
    # mistral model config from huggingface 
    def get_model_config(self, param):
        self.custom_config = MistralConfig(
            vocab_size=param.Vocal,
            hidden_size=param.D_emb,
            intermediate_size=param.d_FF,
            num_hidden_layers=param.N_Layer,
            num_attention_heads=param.N_Head,
            num_key_value_heads=param.KV_Head,
            hidden_act="silu",
            max_position_embeddings=4096 * 32,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=param.use_cache,
            # use_reentrant=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            sliding_window=param.Window,
            attention_dropout=0.0,
        )
        return self.custom_config
    

    
    # mistral model config from mistral-src (github provided)
    def mistral_model_args(self, param):
        raise NotImplementedError("No such implementation yet in ddp/training_utils.py -------------------------------------------> mistral_model_args()")
        # self.args = ModelArgs(
        #     dim = param.D_emb,
        #     n_layers = param.N_Layer,
        #     head_dim = param.d_head,
        #     hidden_dim = param.d_FF,
        #     n_heads = param.N_Head,
        #     n_kv_heads = param.KV_Head,
        #     sliding_window = param.Window,
        #     norm_eps = 1e-5,
        #     vocab_size = param.Vocal,
        #     max_batch_size = 1,
        # )
        # return self.args 
    
    
    # return the model based on the model type
    # if mistral_src == True, then return the model from mistral-src
    # else return the model from huggingface using MistralForCausalLM
    def get_model(self, tokenizer, isfloat16, logger, config = None, mistral_src = False):
        if mistral_src == True:
            print("Loading mistral model from mistral-src")
            print("No such implementation yet")
            raise NotImplementedError
            # self.model = Transformer(self.args)
        else:
            if config is None:
                self.model = MistralForCausalLM(self.custom_config)
            else:
                self.model = MistralForCausalLM(config)
        print("Model created from config")
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        print("Embedding size:", embedding_size)
        print("Tokenizer size:", len(tokenizer))
        if len(tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(tokenizer))
            print("Resized the token embeddings to match the tokenizer size")

        # self.model = self.model.to("cuda:0", dtype= torch.float32)
        print("Model loaded")

        if isfloat16:
            self.model = self.model.half()
            print("Model loaded in half precision")

        print("Total Params v/s one attention layer param:",self.model_size_and_parameters(logger))
        print("Original Model type:",self.model.dtype)

        return self.model
    
    def get_model_from_local(self, local_model_path, logger):
        self.model = MistralForCausalLM.from_pretrained(local_model_path)
        print("Model loaded from local path", local_model_path)
        print("Total Params v/s one attention layer param:",self.model_size_and_parameters(logger))
        print("Original Model type:",self.model.dtype)
        return self.model
    
    def load_model(self, local_model_path=None, logger=None, tokenizer=None, isfloat16=False):
        logger.info(f"{'_'*100}\nLoading the model")
        model = None
        if local_model_path:
            model = self.get_model_from_local(local_model_path=local_model_path, logger= logger)
        else:
            config = self.get_model_config(self.param)    # huggingface mistral config
            model = self.get_model(config = config, tokenizer=tokenizer, isfloat16=isfloat16, logger= logger) # huggingface mistral model
        gpu_usage(logger)
        return model

    def model_size_and_parameters(self, logger = None):
        table = PrettyTable(["Modules", "Parameters"])
        model_size = sum(t.numel() for t in self.model.parameters())
        self.total_params = 0
        self.one_layer_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            self.total_params += params
            if "layers.0" in name:
                self.one_layer_params += params
        # print(table)
        # print(f"MISTRAL model size: {model_size/1000**2:.1f}M parameters")
        # print(f"Total Trainable Params: {self.total_params/10**6:.4f}M")
        # print(f"Total Trainable Params in one layer: {self.one_layer_params/10**6:.4f}M")
        # print("Original Model type:",self.model.dtype)

        if logger:
            logger.info(f'\n{table}')
            logger.info(f"MISTRAL model size: {model_size/1000**2:.1f}M parameters")
            logger.info(f"Total Trainable Params: {self.total_params/10**6:.4f}M")
            logger.info(f"Total Trainable Params in one layer: {self.one_layer_params/10**6:.4f}M")
            logger.info(f"Original Model type:{self.model.dtype}")
            
        return self.total_params, self.one_layer_params
    
    
    

class Dataset_Preprocessing():
    def __init__(self, data_path="/home/dosisiddhesh/SID_DATA_PROCESSED/DATA_2", dataset_batch_size=2, max_seq_length=1024, test_file=None, train_file=None, val_file=None):
        self.dataset_batch_size = dataset_batch_size
        self.max_seq_length = max_seq_length
        self.block_size = max_seq_length
        self.data_path = data_path
        # self.test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'), usecols=['tex','year','month'])
        # self.train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'), usecols=['tex','year','month'])

        # self.test_df['tex'] = self.test_df['tex'].apply(lambda x: os.path.join(data_path, "/".join(x.split('/')[6:])))
        # self.train_df['tex'] = self.train_df['tex'].apply(lambda x: os.path.join(data_path, "/".join(x.split('/')[6:])))

        # print("Test dataset size: ", len(self.test_df))
        # print("Train dataset size: ", len(self.train_df))

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_tokenizer(self, tok_type ,tokenizer_path):
        if tok_type=="mistral_src":
            self.tokenizer = Tokenizer(tokenizer_path)  # mistral-src tokenizer 
        elif tok_type == "debertaV2":
            self.tokenizer =  DebertaV2Tokenizer.from_pretrained(tokenizer_path)  # HF tokenizer
        elif tok_type == "llama":
            self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path) # llama tokenizer
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif tok_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            print("Len of tokenizer before adding special tokens", len(self.tokenizer))
            self.tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                'cls_token': '<cls>',
                                                'sep_token': '<sep>',
                                                'mask_token': '<mask>',
                                                'unk_token': '<unk>',
                                                'bos_token': '<bos>',
                                                'eos_token': '<eos>'
                                            })
            # print the vocab of the tokenizer
            # print("Vocab of the tokenizer: ", self.tokenizer.get_vocab())
            print("Len of tokenizer after adding special tokens", len(self.tokenizer))
        return self.tokenizer
    
    def group_texts(self, examples):

        # old code
        ''' 
            examples = {
                        'input_ids': [[1,2,3],[4,5,6,7,8,9,10,11,12],[13,14,15,...],
            }           'attention_mask': [[1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,...],

            concatenated_examples = { 'input_ids': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,...],
                                    'attention_mask': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,...],
                                    }
            result = {
                        'input_ids': [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,<pad>]],
                        'attention_mask': [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,0]]
                    }

        '''
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Determine the total length after concatenation
        total_length = len(concatenated_examples[list(examples.keys())[0]])     # len(concatenated_examples['input_ids'])
        num_chunks = total_length // self.block_size
        # Calculate the remainder
        remainder = total_length % self.block_size
        
        # Add padding if there is a remainder and the remainder is less than half the block size #IMPORTANT
        if remainder != 0 and 2 * remainder>= self.block_size:
            num_chunks += 1
            padding_length = self.block_size - remainder
            for k in concatenated_examples.keys():
                concatenated_examples[k] += [self.tokenizer.pad_token_id] * padding_length
        # now the length of the concatenated_examples is a multiple of block_size
                

        # # add the num_chunks to the concatenated_examples
        #  = num_chunks

        # Split by chunks of block_size.
        result = { 
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }

        result['num_chunks'] = [num_chunks] + [-1]*(len(result['input_ids'])-1)
        

        '''
            examples = {
                        'input_ids': [[1,2,3],[4,5,6,7,8,9,10,11,12],[13,14,15,...],
            }           'attention_mask': [[1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,...],


            result = {
                        'input_ids': [[1,2,3,<pad>],[4,5,6,7],[8,9,10,11],[13,14,15,<pad>]],
                        'attention_mask': [[1,1,1,0],[1,1,1,1],[1,1,1,1],[1,1,1,0]]
                    }  # chuck with one [12] token is dropped as the chunck length is less than block_size//2
        '''

        # new code
        # no need to write new code as we just change the batch size = 1
        # we can use the old code
        # but in the old code if the batch size is more than one then all the tokens are concatenated and padding is done at end only.
        return result


    def tokenize_function(self, examples):
        return self.tokenizer(                    
                    examples["text"],
                    # truncation=True,
                    # padding=True,
                    # max_length=self.max_seq_length, #8*1024,
                    # return_tensors="pt",
                    # return_token_type_ids=False,
                    # return_length=True,
            )#.to('cuda:0')
    
    def tokenize_function2(self, examples):
        return self.tokenizer(                    
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.max_seq_length, #8*1024,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    # return_length=True,
            )#.to('cuda:0')
    def convert_to_hf_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        # return dataset.map(self.tokenize_function, batched=True, remove_columns=dataset.column_names)
        my_dataset = dataset.map(self.tokenize_function2, 
                           batched=True, 
                           remove_columns=dataset.column_names, 
                           batch_size=self.dataset_batch_size, 
                           num_proc=8)
        print(my_dataset)
        print(len(my_dataset[0]['input_ids']))
        print(len(my_dataset[1]['input_ids']))
        print(len(my_dataset[2]['input_ids']))
        print(len(my_dataset[3]['input_ids']))


        return my_dataset
    
    # No use of this function in the final training flow, only used when the entire data is in a single csv file
    def generate_dataset(self, data_path, row_percent=None, eval_frac=0.1):
        #Setting initial value of the counter to zero 
        rowcount  = 0
        self.data_path = data_path
        #iterating through the whole file 
        for row in open(self.data_path): 
            rowcount+= 1
        #printing the result 
        print("Number of lines present:-", rowcount)
        row_to_read = rowcount if row_percent is None else int(rowcount * row_percent / 100)
        print("loading sample dataset of size ", row_to_read)

        dataframe = pd.read_csv(self.data_path, nrows=row_to_read)
        dataframe = dataframe.dropna()
        print("Total number of rows after dropping NaN: ", len(dataframe))
        df_eval = dataframe.sample(frac=eval_frac, random_state=42)
        dataframe = dataframe.drop(df_eval.index) 
        print("size of dataframe in MB: ", sys.getsizeof(dataframe)/1000000)
        print("Train dataset size: ", len(dataframe))
        print("Validation dataset size: ", len(df_eval))
        print("Train dataset columns: ", dataframe.columns)
        print("Validation dataset columns: ", df_eval.columns)
        self.train_dataset = self.convert_to_hf_dataset(dataframe)
        self.val_dataset = self.convert_to_hf_dataset(df_eval)
        del dataframe, df_eval

    # debug mode will not load the bbl file
    def create_dataset_from_files(self, logger, file_names, month, year, bbl_file_name, bbl_step = 1, isDebug=False):
        '''
        isDebug: if True, then the bbl file will not be loaded, default is False
        bbl_step: the step size to load the bbl file, default is 1, which means all the bbl files will be loaded
        '''
        latex_corpus = []
        print("Reading LaTeX files")
        for file_path in file_names:
            try:
                with open(file_path, 'r', encoding='utf-8') as latex_file:
                    latex_corpus.append(latex_file.read())
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as latex_file:
                    latex_corpus.append(latex_file.read())

        if not latex_corpus:
            logger.error(f"No LaTeX files found in the Month and Year: {month} {year}")
            print("No latex files found")

    
        st = time.time()
        logger.info(f"Encoding tokens started...")
        # created a dataframe from a list of latex data
        df_tex = pd.DataFrame(latex_corpus, columns=['text'])
        if isDebug == False:
            df_bbl = pd.read_csv(bbl_file_name , usecols=['bbl'])
            df_bbl.rename(columns={'bbl': 'text'}, inplace=True)
            df = pd.concat([df_tex, df_bbl], ignore_index=True)
            del df_bbl
        else:
            df = df_tex

        del df_tex

        dataset = Dataset.from_pandas(df)

        del latex_corpus

        st = time.time()
        tokenized_dataset = dataset.map(self.tokenize_function, 
                                    batched=True,
                                    batch_size=4,
                                    remove_columns=['text',
                                                    # '__index_level_0__'
                                                    ],
                                    num_proc=16,
                                )
        print("tokenized_dataset created")

        # convert the tokenized dataset to a list of lists
        latex_corpus_tokenized = tokenized_dataset['input_ids']
        logger.info(f"PAPERS: {len(latex_corpus_tokenized)}, TOTAL TOKENS: {sum([len(x) for x in latex_corpus_tokenized])}")
        logger.info(f"Size of latex_corpus <tokenized>   {sys.getsizeof(latex_corpus_tokenized)}")
        del latex_corpus_tokenized

        et = time.time()
        msg = f"tokenized_dataset created and took {(et - st)/60 :.2f} minutes"
        print(msg)
        print(tokenized_dataset)
        logger.info(msg)
        logger.info(tokenized_dataset)


        lm_datasets = tokenized_dataset.map(
                                        self.group_texts,
                                        batch_size=1,
                                        batched=True,
                                        num_proc=16,
                                )
        msg = f"lm_datasets created and took {(time.time() - et)/60 :.2f} minutes"
        print(msg)
        print(lm_datasets)
        logger.info(msg)
        logger.info(lm_datasets)

        # remove the 'num_chunks' column from the dataset
        num_chunks = lm_datasets['num_chunks']
        # remove all the values of -1 from the list
        # chunks = []
        total_num_chunks = 0
        for chunk in num_chunks:
            if chunk != -1:
                total_num_chunks += chunk
                # chunks.append(chunk)

        print("Total number of chunks: ", total_num_chunks)
        logger.info(f"Total number of chunks: {total_num_chunks}")

        lm_datasets.remove_columns('num_chunks')
        
        # pickle.dump(lm_datasets, open(f"{dest_path}/{yr}_{month}_lm_datasets.pkl", "wb"))
        # print("lm_datasets dumped to pickle file")
        # logger.info("lm_datasets dumped to pickle file")

        return lm_datasets
        

    def generate_dataset_from_files(self, month, year, logger=None, test_percent=0, train_percent=1, val_percent=1, isDebug=False):
        '''
        test_percent: the percentage of the test files to be used, default is 0
        train_percent: the percentage of the train files to be used, default is 1
        val_percent: the percentage of the validation files to be used, default is 1
        isDebug: if True, then the bbl file will not be loaded, default is False
        '''
        yymm = int(f'{year}{month}' if month>9 else f'{year}0{month}')
        bbl_yymm = f"0{year}{f'0{month}' if month<10 else f'{month}'}" if year<10 else f"{year}{f'0{month}' if month<10 else f'{month}'}"
        

        file_names = self.test_df[(self.test_df['year']==year) & (self.test_df['month']==yymm)]['tex'].to_list()
        bbl_test_file_name = os.path.join(self.data_path, 'TEST_BBL',f'bbl_split_test_{bbl_yymm}.csv')

        val_file_names = file_names[:int(len(file_names)*val_percent)]
        test_file_names = file_names[int(len(file_names)*val_percent):int(len(file_names)*val_percent)+int(len(file_names)*test_percent)]

        print("-------------------------------------")
        print("Total test file names: ", len(file_names))
        print("Val file names: ", len(val_file_names))
        print("Test file names: ", len(test_file_names))
        print("-------------------------------------")


        if len(val_file_names) == 0:
            print("No validation files create")
        else:
            print(f"Validation dataset size: {len(val_file_names)} for year: {year} and month: {month}")
            logger.info(f"Validation dataset size: {len(val_file_names)} for year: {year} and month: {month}")
            self.val_dataset = self.create_dataset_from_files(logger, val_file_names, month, year, bbl_test_file_name, bbl_step = 1, isDebug=isDebug)    
        
        if len(test_file_names) == 0:
            print("No test files create")
        else:
            print(f"Test dataset size: {len(test_file_names)} for year: {year} and month: {month}")
            logger.info(f"Test dataset size: {len(test_file_names)} for year: {year} and month: {month}")
            self.test_dataset = self.create_dataset_from_files(logger, test_file_names, month, year, bbl_test_file_name, bbl_step = 1, isDebug=isDebug)    
        
        

         
        file_names = self.train_df[(self.train_df['year']==year) & (self.train_df['month']==yymm)]['tex'].to_list()
        train_file_names = file_names[:int(len(file_names)*train_percent)]
        print("-------------------------------------")
        print("Total Train file names: ", len(file_names))
        print("Final Train file names: ", len(train_file_names))
        print("-------------------------------------")
        bbl_file_name = os.path.join(self.data_path, 'TRAIN_BBL',f'bbl_split_train_{bbl_yymm}.csv')
        
        

        if len(train_file_names) == 0:
            print("No train files create")
        else:
            print(f"Train dataset size: {len(train_file_names)} for year: {year} and month: {month}")
            logger.info(f"Train dataset size: {len(train_file_names)} for year: {year} and month: {month}")
            self.train_dataset = self.create_dataset_from_files(logger, train_file_names, month, year, bbl_file_name, bbl_step = 1, isDebug=isDebug)




        

    def get_train_dataset(self, local_path=None, sample_size=None, batch_size=1):
        if local_path:
            # load the dataset from pickle file
            # path = '/home/dosisiddhesh/MISTRAL_EXP/data/00_01_lm_datasets.pkl'
            with open(local_path, 'rb') as f:
                self.train_dataset = pickle.load(f)
                # self.train_dataset = Dataset.from_dict(pickle.load(f))
                if sample_size:
                    self.train_dataset = self.train_dataset.select(range(sample_size*batch_size))
                    # save the dataset to a pickle file
                    # pickle.dump(self.train_dataset, open(os.path.join('/home/dosisiddhesh/latex_model/data',f"sample_train_{sample_size}.pkl" ),"wb+"))
                    pickle.dump(self.train_dataset, open(os.path.join(os.path.split(local_path)[0],f"sample_train_{sample_size}.pkl" ),"wb+"))

            return self.train_dataset
        if self.train_dataset is None:
            print("No train dataset found")
        return self.train_dataset
    
    def get_val_dataset(self, local_path=None, sample_size=None,  batch_size=1):
        if local_path:
            # load the dataset from pickle file
            # path = '/home/dosisiddhesh/MISTRAL_EXP/data/00_01_lm_datasets.pkl'
            with open(local_path, 'rb') as f:
                self.val_dataset = pickle.load(f)
                # self.val_dataset = Dataset.from_dict(pickle.load(f))
                if sample_size:
                    self.val_dataset = self.val_dataset.select(range(sample_size))
                    # save the dataset to a pickle file
                    # pickle.dump(self.val_dataset, open(os.path.join('/home/dosisiddhesh/latex_model/data',f"sample_val_{sample_size}.pkl" ),"wb+"))
                    pickle.dump(self.val_dataset, open(os.path.join(os.path.split(local_path)[0],f"sample_val_{sample_size}.pkl"),"wb+"))
            return self.val_dataset
        if self.val_dataset is None:
            print("No val dataset found")
        return self.val_dataset
    
    def get_test_dataset(self, local_path=None, sample_size=None,  batch_size=1):
        if local_path:
            # load the dataset from pickle file
            # path = '/home/dosisiddhesh/MISTRAL_EXP/data/00_01_lm_datasets.pkl'
            with open(local_path, 'rb') as f:
                self.test_dataset = pickle.load(f)
                # self.test_dataset = Dataset.from_dict(pickle.load(f))
                if sample_size:
                    self.test_dataset = self.test_dataset.select(range(sample_size))
                    # save the dataset to a pickle file
                    # pickle.dump(self.test_dataset, open(os.path.join('/home/dosisiddhesh/latex_model/data',f"sample_test_{sample_size}.pkl"),"wb+"))
                    pickle.dump(self.test_dataset, open(os.path.join(os.path.split(local_path)[0],f"sample_test_{sample_size}.pkl"),"wb+"))
            return self.test_dataset
        if self.test_dataset is None:
            print("No test dataset found")
        return self.test_dataset

    def get_train_dataloader(self, collate_fn=None, batch_size=2):
        self.collate_fn = collate_fn if collate_fn is not None else DataCollatorForLanguageModeling(self.tokenizer, mlm=False, return_tensors="pt" )
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True)
        return self.train_dataloader
    
    def get_val_dataloader(self, collate_fn=None, batch_size=2):
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True)
        return self.val_dataloader


#  rsync -ra -e 'ssh -p 2020' --info=progress2  SID_DATA_PROCESSED dosisiddhesh@10.0.62.205:/home/dosisiddhesh/