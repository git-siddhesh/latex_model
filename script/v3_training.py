#  CUDA_VISIBLE_DEVICES=0 python v3_training.py  --float16 --layer 13 --seq_len 2048 --batch_size 32 --max_grad_norm 0.5 --test
#  CUDA_VISIBLE_DEVICES=0 python v2_mp_mistral.py  --float16 --layer 13 --seq_len 2048 --batch_size 32 --max_grad_norm 0.5 --checkpoint /home/dosisiddhesh/HITESH_LODWAL/llm_guru_hindi/model2/latex/main3_ep_1_lr_0.0006_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_13_N_Head_32_KV_Head_8_W_4096 --start_sample_file_index 13
#  CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 6 --seq_len 2048 --batch_size 32 --max_grad_norm 0.5 --test 
# eval loss is working with above call

# current training model : float 32, without quantization, without gradient checkpointing, 
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 5 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9


#%% Experiement : fp16_full_eval = True, in training arguments
# CUDA_VISIBLE_DEVICES=2 python v3_training.py --layer 5 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --float16 --test
# conda activate env_mist
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 2 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --test 


import os

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

import argparse
parser = argparse.ArgumentParser()

# original args
parser.add_argument("--device", type=str, default="0", help="cuda device number", choices=["0", "1", "2", "3"])
parser.add_argument("--layer", type=int, default=12, help="number of layers")
parser.add_argument("--seq_len", type=int, default=2*1024, help="sequence length")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--float16", action="store_true", help="use float16")
parser.add_argument("--adafactor", action='store_true', help="use adafactor")
parser.add_argument("--enb_grad_checkpoint", action='store_true', help="disable use cache in model config and enable gradient checkpointing")
parser.add_argument("--data_percent", type=float, default=0.001, help="data row percent")
parser.add_argument("--vocab", type=int, default=30000, help="vocab size")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path incase not loading the model from saved directory")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--start_sample_file_index", type=int, default=1, help="start_sample_file_index")
# load local model 
parser.add_argument("--local_model_path", default=None, help="load the model from the local directory in case in not loading from checkpoint")

args = parser.parse_args()
print( ',  '.join([i[0]+':'+str(i[1]) for i in args._get_kwargs()]))

from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

import wandb
wandb.login()
os.environ["WANDB_PROJECT"]="latex_fp32"+date
WANDB_PROJECT="latex_training"+date
wandb_run_name = f"run_latex_fp32_{args.layer}_{args.seq_len}_{args.batch_size}_{args.max_grad_norm}_{args.vocab}_{timestamp}"

#%% SET UP THE PATH
ta_logging_dir="./logs_grad_clip"
ta_report_to="wandb"
ta_run_name = wandb_run_name

root_dir = '/home/iitgn_cse/latex_model'
data_path = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/DATA_TKNZD_10-4-24'
DATA_PATH_PICKEL = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/DATA_TKNZD_10-4-24'
TOKENIZER_HF_ST_PATH = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/hf_tokenizer_2.0%_30000_without_whitespace_pretokenizer_79372_outof_3968648'
ROOT_LOG_DIR = os.path.join(root_dir,'log_main_fp32'+date)
MODEL_ROOT_DIR = os.path.join(root_dir,'model_main_fp32'+date)

os.makedirs(ROOT_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ['WANDB_DISABLED'] = 'true'


import logging
import sys
import time
import tqdm
import torch
from pathlib import Path
# from evaluate import load
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    # EarlyStoppingCallback, 
)
# import evaluate
# from mistral.tokenizer import Tokenizer
# from mistral.model import Transformer, ModelArgs
from training_utils import (
    Parameter,
    MyModel,
    Dataset_Preprocessing,
    HyperParams
)

from pynvml import *
# from optimum.intel import OVConfig, OVTrainer, OVModelForCausalLM, OVTrainingArguments

# from optimum.intel.openvino.quantization 
# from optimum.intel.openvino.modeling import OVModel

# ___________________________________________________________________________________________________________________________
# *********************** Local code, model and data path ***********************************************************************************************************

# metric = load("perplexity")

# tokenizer_path_sentence_piece_for_mistral_src = '/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model'
# tokenizer_path_hf_debertv2 = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_hf.model"
# tokenizer_path_llama = "hf-internal-testing/llama-tokenizer" #llama


def gpu_usage(logger):
    def print_gpu_utilization(logger):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied from nvmlInit: {info.used//1024**2} MB.")
        logger.info(f"GPU memory occupied from nvmlInit: {info.used//1024**2} MB.")

    print("+----------------------------------------------------------------------------------+")
    a,b = torch.cuda.mem_get_info()
    gpu_mem_usage = (b-a)/(2**20)
    print(f"GPU memory usage before cleaning cache: {gpu_mem_usage:.2f} MB")
    logger.info(f"GPU memory usage before cleaning cache: {gpu_mem_usage:.2f} MB")

    torch.cuda.empty_cache()
    
    a,b = torch.cuda.mem_get_info()
    gpu_mem_usage = (b-a)/(2**20)
    print(f"GPU memory usage after cleaning cache: {gpu_mem_usage:.2f} MB")
    logger.info(f"GPU memory usage after cleaning cache: {gpu_mem_usage:.2f} MB")
    print_gpu_utilization(logger)
    print("+----------------------------------------------------------------------------------+")

# ___________________________________________________________________________________________________________________________
# *********************** @QAT ***********************************************************************************************************

# ov_config = OVConfig(save_onnx_model=True)

# print("------------------- OV Config -------------------")
# print(ov_config)
# logger.info(ov_config)


# In[]: __________________________________________________________________________________________________
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
D_emb = 4*1024
Vocal = args.vocab
d_head = 128# + 64
d_FF = 14336
N_Layer = args.layer
N_Head = 32
KV_Head = 8
Window = 4096 #8192
# data_row = 100
value = [D_emb,Vocal,d_head,d_FF,N_Layer,N_Head,KV_Head,Window]
#**************************************************************************************************
param = Parameter("Mistral", value, use_cache= not args.enb_grad_checkpoint)
hp = HyperParams(
    epoch=1, 
    learning_rate=2e-5, 
    model_id="latex/main_fp32_"+date,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine", #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
    BATCH_SIZE=args.batch_size,
    tokenizer_batch_size=16,
    eval_steps=10 if args.test else 4000, # Adjust as needed1
    logging_steps=10 if args.test else 4000,  # Adjust as needed
    save_steps=10 if args.test else 4000,
    save_total_limit = 3,
    max_seq_length=int(args.seq_len),
)

model_obj = MyModel(model_id=hp.model_id, hp=hp, param=param)
print(model_obj.model_name)
log_file_name = f"{ROOT_LOG_DIR}/log_{model_obj.model_name.split('/')[-1]}_{timestamp}.log"
logging.basicConfig(level=logging.NOTSET, filename=log_file_name, filemode="w", format="%(asctime)-15s %(name)-10s %(levelname)-8s %(message)s")

logging.getLogger("lightning.pytorch").setLevel(logging.NOTSET)
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))
# add a line to seperate current log from prev
logger.info("\n\n______________________________________________________________________________________")
logger.info("+++++++++++++++++++++++++++++++++++++ NEW RUN ++++++++++++++++++++++++++++++++++++++++\n")
logger.info(',  '.join([i[0]+':'+str(i[1]) for i in args._get_kwargs()]))
logger.info(f"D_emb: {D_emb}, Vocal: {Vocal}, d_head: {d_head}, d_FF: {d_FF}, N_Layer: {N_Layer}, N_Head: {N_Head}, KV_Head: {KV_Head}, Window: {Window}")
# logger.info(f"Training data rows: {data_row}")
logger.info(f"Epoch: {hp.epochs}, Learning rate: {hp.learning_rate}, Weight decay: {hp.weight_decay}, Warmup steps: {hp.warmup_steps}, LR scheduler type: {hp.lr_scheduler_type}, Batch size: {hp.BATCH_SIZE}, Eval steps: {hp.eval_steps}, Logging steps: {hp.logging_steps}, Save steps: {hp.save_steps}, Save total limit: {hp.save_total_limit}, Max seq length: {hp.max_seq_length}")

#____________________________________________________________________________________________________________________________
# In[]: preparing the dataset ***********************************************************************************************
dataset_obj = Dataset_Preprocessing(data_path, dataset_batch_size=hp.tokenizer_batch_size, max_seq_length=hp.max_seq_length)
tokenizer = dataset_obj.load_tokenizer(tok_type="hf", tokenizer_path=TOKENIZER_HF_ST_PATH)


def get_model(local_model_path=None):
    print("Loading model...")
    model = None
    if local_model_path:
        model = model_obj.get_model_from_local(local_model_path=local_model_path, logger= logger)
    else:
        config = model_obj.get_model_config(param)    # huggingface mistral config
        model = model_obj.get_model(config = config, tokenizer=tokenizer, isfloat16=args.float16, logger= logger) # huggingface mistral model
    gpu_usage(logger)
    return model

# metric = evaluate.load("accuracy")
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # preds have the same shape as the labels, after the argmax(-1) has been calculated
#     # by preprocess_logits_for_metrics but we need to shift the labels
#     labels = labels[:, 1:].reshape(-1)
#     preds = preds[:, :-1].reshape(-1)
#     return metric.compute(predictions=preds, references=labels)
#____________________________________________________________________________________________________________________________
# In[]: Trainning the model using optimum transformer trainer *************************************************************************************************
# training_args = OVTrainingArguments(
training_args = TrainingArguments(
    # distillation_weight = 0.5, # default 0.5
    # distillation_temperature = 0.2 # default 0.2
    remove_unused_columns=True,
    output_dir=os.path.join(MODEL_ROOT_DIR, model_obj.model_name),  # Change to your desired output directory
    overwrite_output_dir=True,
    # per_device_train_batch_size=hp.BATCH_SIZE,  # Adjust as needed current 1
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=hp.eval_steps, # Adjust as needed1
    logging_steps=hp.logging_steps,  # Adjust as needed
    gradient_accumulation_steps=hp.BATCH_SIZE,
    num_train_epochs=hp.epochs,  # Adjust as needed
    weight_decay=hp.weight_decay,
    warmup_steps=hp.warmup_steps,
    lr_scheduler_type=hp.lr_scheduler_type,
    learning_rate=hp.learning_rate,
    load_best_model_at_end=True, 
    save_steps=hp.save_steps,  # Adjust as needed
    # fp16=True if args.float16 else False,
    # fp16 = True,
    optim='adafactor' if args.adafactor else 'adamw_torch',
    # optim="adamw_bnb_8bit"
    gradient_checkpointing=args.enb_grad_checkpoint,
    save_total_limit=hp.save_total_limit,  # Adjust as needed
    logging_dir=ta_logging_dir,
    report_to=ta_report_to,
    run_name = ta_run_name,
    # resume_from_checkpoint=os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model2", model_obj.model_name)
    resume_from_checkpoint=args.checkpoint,
    max_grad_norm = args.max_grad_norm,
    # fp16_full_eval = True,

)



model = get_model(local_model_path = args.local_model_path)
# checkpoint = '/home/dosisiddhesh/MISTRAL_EXP/model2/mistral/dummy_ep_1_lr_0.0006_linear_weight_decay_0.1_warmup_steps_100'
# trainer = OVTrainer(
trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    # compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # ov_config=ov_config,
    # task="text-generation",
)

# #%% Main training loop
# for i in range(args.start_sample_file_index, 55):
for year in range(24):
    for month in range(1, 13):
        print("------------------------------------------------------------------------------------------")
        print(f"Training for {year}-{month}")
        logger.info(f"Training for {year}-{month}")
        print("------------------------------------------------------------------------------------------")
        val_local_pickel_path = os.path.join(DATA_PATH_PICKEL, f"val_{year}_{month}_datasets.pkl")
        train_local_pickel_path = os.path.join(DATA_PATH_PICKEL, f"train_{year}_{month}_datasets.pkl")
        print(f"val_local_pickel_path: {val_local_pickel_path}")
        print(f"train_local_pickel_path: {train_local_pickel_path}")
        logger.info(f"val_local_pickel_path: {val_local_pickel_path}")
        logger.info(f"train_local_pickel_path: {train_local_pickel_path}")
        
        if not os.path.exists(val_local_pickel_path):
            print(f"File not found: {val_local_pickel_path}")
            logger.info(f"File not found: {val_local_pickel_path}")
            exit()
        if not os.path.exists(train_local_pickel_path):
            print(f"File not found: {train_local_pickel_path}")
            logger.info(f"File not found: {train_local_pickel_path}")
            exit()

        trainer.train_dataset = dataset_obj.get_train_dataset(local_path=train_local_pickel_path, sample_size=hp.eval_steps*hp.BATCH_SIZE*3 if args.test else None)
        trainer.eval_dataset = dataset_obj.get_val_dataset(local_path=val_local_pickel_path, sample_size=hp.eval_steps if args.test else None)
        # train_dataset.select(range(10))
        print(trainer.train_dataset)
        print(trainer.eval_dataset)
        logger.info(trainer.train_dataset)
        logger.info(trainer.eval_dataset)


        # trainer.train_dataset = train_dataset
        # trainer.eval_dataset = val_dataset

        try:
            # Train the model while applying quantization
            train_result = None
            if args.checkpoint:
                train_result = trainer.train(resume_from_checkpoint=args.checkpoint)
            else:
                train_result = trainer.train()
            print(f"***** Train results ***** {train_result}")
            logger.info(f"***** Train results ***** {train_result}")
            gpu_usage(logger)

        except Exception as e: 
            print("Error occured while training the model ???????????????????????????????????")
            logger.error("Error occured while training the model ???????????????????????????????????")
            print(e)
            logger.error(e)
            exit()
        try:

            metrics = trainer.evaluate()
            print(f"***** Eval results ***** {metrics}")
            logger.info(f"***** Eval results ***** {metrics}")
        except Exception as e:
            logger.error("Error occured while training the model ???????????????????????????????????")
            print(e)
            logger.error(e)

        try:
            # Export the quantized model to OpenVINO IR format and save it
            trainer.save_model()
            print("Model saved")
        except Exception as e:
            print("Error occured while saving the model ???????????????????????????????????")
            logger.error("Error occured while saving the model ???????????????????????????????????")
            print(e)
            logger.error(e)
        gpu_usage(logger)

# In[]: ___________________________________________________________________________________________________________________
# last working config
'''
D_emb = 4096
Vocal = 30000
d_head = 128
d_FF = 7168 #14336
N_Layer = 4
N_Head = 32
KV_Head = 8
Window = 4096 #8192
data_row = 100
value = [D_emb,Vocal,d_head,d_FF,N_Layer,N_Head,KV_Head,Window]
#**************************************************************************************************
param = Parameter("Mistral", value)
hp = HyperParams(
    epoch=1, 
    learning_rate=6e-4, 
    model_id="mistral/dummy",
    weight_decay=0.1,  
    warmup_steps=50,
    lr_scheduler_type="linear", #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
    BATCH_SIZE=4,
    tokenizer_batch_size=16,
    eval_steps=50, # Adjust as needed1
    logging_steps=50,  # Adjust as needed
    save_steps=200,
    save_total_limit = 1,
    max_seq_length=int(1024*2),
)
'''

#----------------------------------------------
# if checkpoint:
#     print(f"Loading the model checkpoint for {year}-{month}")
#     logger.info(f"Loading the model checkpoint for {year}-{month}: {checkpoint}")
#     model = OVModelForCausalLM.from_pretrained(checkpoint, export=True)
#     print("Model loaded")
#     logger.info("Model loaded")
#     gpu_usage(logger)
#     checkpoint = os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model2", model_obj.model_name)
# else:
#     model = get_model()

#-----------------------------------------------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 v2_mp_mistral.py --layer 10 --seq_len 2048 --float16 