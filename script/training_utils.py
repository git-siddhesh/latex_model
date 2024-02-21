
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

code_path = "/home/dosisiddhesh/MISTRAL_EXP/mistral-src"
data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/latex.csv"
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/model/mistral-7B-v0.1")  # model and tokenizer location
tokenizer_path = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model" # sentencepiece tokenizer path
sys.path.append(code_path)  # append the path where mistral-src was cloned
from mistral.tokenizer import Tokenizer
# from mistral.model import Transformer, ModelArgs

class Parameter:
    def __init__(self, name, value, use_cache=True):
        self.name = name
        self.D_emb,self.Vocal,self.d_head,self.d_FF,self.N_Layer,self.N_Head,self.KV_Head,self.Window = value
        self.use_cache = use_cache

class HyperParams:
    def __init__(self, epoch = 1, learning_rate = 3e-4, model_id = "mistral", **kwargs):
        self.learning_rate = learning_rate
        self.epochs = epoch
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
        self.eval_batch_size=kwargs.get('eval_batch_size',2) #2
        self.eval_frac=kwargs.get('eval_frac', 0.1)
        self.max_seq_length=kwargs.get('max_seq_length', 1024)


    
class MyModel(Parameter, HyperParams):
    def __init__(self, model_id="mistral", hp=None):
        self.model_id = model_id
        self.args = None
        if hp is not None:
            self.model_name = f"{self.model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_{hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"
    
    def get_model_name(self,hp):
        self.model_name = f"{self.model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_{hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"

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
            pad_token_id=None,
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
    
    # def get_model_checkpoint(self, checkpoint_path, mistral_src = False):
    #     if mistral_src == True:
    #         print("No such implementation yet")
    #         exit()
    #         return None

    #     self.model = 
    #     return self.model
    
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
        print(table)
        print(f"MISTRAL model size: {model_size/1000**2:.1f}M parameters")
        print(f"Total Trainable Params: {self.total_params/10**6:.4f}M")
        print(f"Total Trainable Params in one layer: {self.one_layer_params/10**6:.4f}M")
        print("Original Model type:",self.model.dtype)

        if logger:
            logger.info(table)
            logger.info(f"MISTRAL model size: {model_size/1000**2:.1f}M parameters")
            logger.info(f"Total Trainable Params: {self.total_params/10**6:.4f}M")
            logger.info(f"Total Trainable Params in one layer: {self.one_layer_params/10**6:.4f}M")
            logger.info("Original Model type:",self.model.dtype)
            
        return self.total_params, self.one_layer_params
    
    
    

class Dataset_Preprocessing():
    def __init__(self, data_path="/home/dosisiddhesh/SID_DATA_PROCESSED/DATA_2", dataset_batch_size=2, max_seq_length=1024 ):
        self.dataset_batch_size = dataset_batch_size
        self.max_seq_length = max_seq_length
        self.block_size = max_seq_length
        self.data_path = data_path
        self.test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'), usecols=['tex','year','month'])
        self.test_df['tex'] = self.test_df['tex'].apply(lambda x: os.path.join(data_path, "/".join(x.split('/')[6:])))

        self.train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'), usecols=['tex','year','month'])
        self.train_df['tex'] = self.train_df['tex'].apply(lambda x: os.path.join(data_path, "/".join(x.split('/')[6:])))

        print("Test dataset size: ", len(self.test_df))
        print("Train dataset size: ", len(self.train_df))

    def load_tokenizer(self, tok_type ,tokenizer_path):
        if tok_type=="mistral_src":
            self.tokenizer = Tokenizer(tokenizer_path)  # mistral-src tokenizer 
        elif tok_type == "debertaV2":
            self.tokenizer =  DebertaV2Tokenizer.from_pretrained(tokenizer_path)  # HF tokenizer
        elif tok_type == "llama":
            self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path) # llama tokenizer
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif tok_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                'cls_token': '<cls>',
                                                'sep_token': '<sep>',
                                                'mask_token': '<mask>',
                                                'unk_token': '<unk>',
                                                'bos_token': '<bos>',
                                                'eos_token': '<eos>'
                                            })
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

    def convert_to_hf_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        # return dataset.map(self.tokenize_function, batched=True, remove_columns=dataset.column_names)
        my_dataset = dataset.map(self.tokenize_function, 
                           batched=True, 
                           remove_columns=dataset.column_names, 
                           batch_size=self.dataset_batch_size, 
                           num_proc=8)

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


    def create_dataset_from_files(self, logger, file_names, month, year, bbl_file_name):
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
            return None

    
        st = time.time()
        logger.info(f"Encoding tokens started...")
        # created a dataframe from a list of latex data
        df_tex = pd.DataFrame(latex_corpus, columns=['text'])

        df_bbl = pd.read_csv(bbl_file_name , usecols=['bbl'])
        df_bbl.rename(columns={'bbl': 'text'}, inplace=True)
        df = pd.concat([df_tex, df_bbl], ignore_index=True)
        del df_tex, df_bbl

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
        

    def generate_dataset_from_files(self, month, year, logger=None):
        yymm = int(f'{year}{month}' if month>9 else f'{year}0{month}')
        bbl_yymm = f"0{year}{f'0{month}' if month<10 else f'{month}'}" if year<10 else f"{year}{f'0{month}' if month<10 else f'{month}'}"


        file_names = self.test_df[(self.test_df['year']==year) & (self.test_df['month']==yymm)]['tex'].to_list()
        print(f"Test dataset size: {len(file_names)} for year: {year} and month: {month}")
        bbl_file_name = os.path.join(self.data_path, 'TEST_BBL',f'bbl_split_test_{bbl_yymm}.csv')
        self.val_dataset = self.create_dataset_from_files(logger, file_names, month, year, bbl_file_name)
        
        
        file_names = self.train_df[(self.train_df['year']==year) & (self.train_df['month']==yymm)]['tex'].to_list()
        print(f"Train dataset size: {len(file_names)} for year: {year} and month: {month}")
        bbl_file_name = os.path.join(self.data_path, 'TRAIN_BBL',f'bbl_split_train_{bbl_yymm}.csv')
        self.train_dataset = self.create_dataset_from_files(logger, file_names, month, year, bbl_file_name)



        

    def get_train_dataset(self, local_path=None):
        if local_path:
            # load the dataset from pickle file
            path = '/home/dosisiddhesh/MISTRAL_EXP/data/00_01_lm_datasets.pkl'
            with open(path, 'rb') as f:
                self.train_dataset = pickle.load(f)
            return self.train_dataset
        return self.train_dataset
    
    def get_val_dataset(self, local_path=None):
        if local_path:
            # load the dataset from pickle file
            path = '/home/dosisiddhesh/MISTRAL_EXP/data/00_01_lm_datasets.pkl'
            with open(path, 'rb') as f:
                self.val_dataset = pickle.load(f)
            return self.val_dataset
        return self.val_dataset

    def get_train_dataloader(self, collate_fn=None, batch_size=2):
        self.collate_fn = collate_fn if collate_fn is not None else DataCollatorForLanguageModeling(self.tokenizer, mlm=False, return_tensors="pt" )
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True)
        return self.train_dataloader
    
    def get_val_dataloader(self, collate_fn=None, batch_size=2):
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True)
        return self.val_dataloader


#  rsync -ra -e 'ssh -p 2020' --info=progress2  SID_DATA_PROCESSED dosisiddhesh@10.0.62.205:/home/dosisiddhesh/