import os
#__________________________________________________________________________________________________
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
import wandb
wandb.login()
os.environ["WANDB_PROJECT"]="Misral"
WANDB_PROJECT="Misral_training"
wandb_run_name = "run1"



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
parser.add_argument("--seq_len", type=int, default=4*1024, help="sequence length")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--float16", action="store_true", help="use float16")
parser.add_argument("--adafactor", action='store_true', help="use adafactor")
parser.add_argument("--enb_grad_checkpoint", action='store_true', help="disable use cache in model config and enable gradient checkpointing")
parser.add_argument("--data_percent", type=float, default=0.001, help="data row percent")
parser.add_argument("--vocab", type=int, default=30000, help="vocab size")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")

args = parser.parse_args()
print( ',  '.join([i[0]+':'+str(i[1]) for i in args._get_kwargs()]))



# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["CUDA_LAUNCH_BLOCKING"]='0,1'
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ['WANDB_DISABLED'] = 'true'


import logging
from datetime import datetime
import sys
import time
import tqdm
import torch
from pathlib import Path
from evaluate import load
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    EarlyStoppingCallback, 
)
import evaluate
# from mistral.tokenizer import Tokenizer
# from mistral.model import Transformer, ModelArgs
from training_utils import Parameter, MyModel, Dataset_Preprocessing, HyperParams

from pynvml import *
from optimum.intel import OVConfig, OVTrainer, OVModelForCausalLM, OVTrainingArguments

# from optimum.intel.openvino.quantization 
from optimum.intel.openvino.modeling import OVModel


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
root_log_dir = '/home/dosisiddhesh/MISTRAL_EXP/log/'
logging.basicConfig(level=logging.NOTSET, filename="{}/QAT_log_{}.log".format(root_log_dir, timestamp), filemode="w", format="%(asctime)-15s %(name)-10s %(levelname)-8s %(message)s")

# logging.basicConfig(level=logging.NOTSET ,filename=f"{root_log_dir}QAT_log_{timestamp}.log", filemode="w", format="%(asctime)-15s %(name)-10s %(levelname)-8s %(message)s") 
# configure logging at the root level of Lightning
# logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.NOTSET)
# configure logging on module level, redirect to file
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))





# ___________________________________________________________________________________________________________________________
# *********************** Local code, model and data path ***********************************************************************************************************

# metric = load("perplexity")
code_path = "/home/dosisiddhesh/MISTRAL_EXP/mistral-src"
data_path = '/home/dosisiddhesh/SID_DATA_PROCESSED/DATA_2'

model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/model/mistral-7B-v0.1")  # model and tokenizer location
# tokenizer_path_sentence_piece_for_mistral_src = '/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model'
# tokenizer_path_hf_debertv2 = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_hf.model"
# tokenizer_path_llama = "hf-internal-testing/llama-tokenizer" #llama
tokenizer_path_hf_our = None
if args.vocab == 30000:
    tokenizer_path_hf_our = '/home/dosisiddhesh/MISTRAL_EXP/model/hf_tokenizer_1.0%_30000_new'
elif args.vocab == 50000:
    tokenizer_path_hf_our = '/home/dosisiddhesh/MISTRAL_EXP/model/hf_tokenizer_4.0%_50000_new'


sys.path.append(code_path)  # append the path where mistral-src was cloned


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied from method1: {info.used//1024**2} MB.")
    logger.info(f"GPU memory occupied from method1: {info.used//1024**2} MB.")

def gpu_usage():
    print("+----------------------------------------------------------------------------------+")
    print_gpu_utilization()
    torch.cuda.empty_cache()
    a,b = torch.cuda.mem_get_info()
    gpu_mem_usage = (b-a)/(2**20)
    print(f"GPU memory usage before training: {gpu_mem_usage:.2f} MB")
    logger.info(f"GPU memory usage before training: {gpu_mem_usage:.2f} MB")
    print_gpu_utilization()
    print("+----------------------------------------------------------------------------------+")

# ___________________________________________________________________________________________________________________________
# *********************** @QAT ***********************************************************************************************************

ov_config = OVConfig(save_onnx_model=True)

# print("------------------- OV Config -------------------")
# print(ov_config)
# logger.info(ov_config)


# In[]: __________________________________________________________________________________________________
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
D_emb = 4*1024
Vocal = args.vocab
d_head = 128
d_FF = 14336
N_Layer = args.layer
N_Head = 32
KV_Head = 8
Window = 4096 #8192
data_row = 100
value = [D_emb,Vocal,d_head,d_FF,N_Layer,N_Head,KV_Head,Window]
#**************************************************************************************************
param = Parameter("Mistral", value, use_cache= not args.enb_grad_checkpoint)
hp = HyperParams(
    epoch=1, 
    learning_rate=6e-4, 
    model_id="mistral/main",
    weight_decay=0.1,  
    warmup_steps=100,
    lr_scheduler_type="linear", #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
    BATCH_SIZE=args.batch_size,
    tokenizer_batch_size=16,
    eval_steps=2000, # Adjust as needed1
    logging_steps=2000,  # Adjust as needed
    save_steps=10000,
    save_total_limit = 3,
    max_seq_length=int(args.seq_len),
)

# logger.info(f"args: {args}")
logger.info(',  '.join([i[0]+':'+str(i[1]) for i in args._get_kwargs()]))
logger.info(f"D_emb: {D_emb}, Vocal: {Vocal}, d_head: {d_head}, d_FF: {d_FF}, N_Layer: {N_Layer}, N_Head: {N_Head}, KV_Head: {KV_Head}, Window: {Window}")
logger.info(f"Training data rows: {data_row}")
logger.info(f"Epoch: {hp.epochs}, Learning rate: {hp.learning_rate}, Weight decay: {hp.weight_decay}, Warmup steps: {hp.warmup_steps}, LR scheduler type: {hp.lr_scheduler_type}, Batch size: {hp.BATCH_SIZE}, Eval steps: {hp.eval_steps}, Logging steps: {hp.logging_steps}, Save steps: {hp.save_steps}, Save total limit: {hp.save_total_limit}, Max seq length: {hp.max_seq_length}")


#____________________________________________________________________________________________________________________________
# In[]: preparing the dataset ***********************************************************************************************
dataset_obj = Dataset_Preprocessing(data_path, dataset_batch_size=hp.tokenizer_batch_size, max_seq_length=hp.max_seq_length)
print("Loading tokenizer")

#-----------------------------------------------------------------------------------------------------------------------------
tokenizer = dataset_obj.load_tokenizer(tok_type="hf", tokenizer_path=tokenizer_path_hf_our)

# -----------------------------------------------------------------------------------------------------------------------------

model_obj = MyModel(model_id=hp.model_id, hp=hp)
def get_model():
    print("Loading model...")
    config = model_obj.get_model_config(param)    # huggingface mistral config
    model = model_obj.get_model(config = config, tokenizer=tokenizer, isfloat16=args.float16, logger= logger) # huggingface mistral model
    gpu_usage()
    return model

print(model_obj.model_name)



metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)
#____________________________________________________________________________________________________________________________
# In[]: Trainning the model using optimum transformer trainer *************************************************************************************************
training_args = OVTrainingArguments(
    # distillation_weight = 0.5, # default 0.5
    # distillation_temperature = 0.2 # default 0.2
    remove_unused_columns=True,
    output_dir=os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model2", model_obj.model_name),  # Change to your desired output directory
    overwrite_output_dir=True,
    per_device_train_batch_size=hp.BATCH_SIZE,  # Adjust as needed
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=hp.eval_steps, # Adjust as needed1
    logging_steps=hp.logging_steps,  # Adjust as needed
    # gradient_accumulation_steps=8,
    num_train_epochs=hp.epochs,  # Adjust as needed
    weight_decay=hp.weight_decay,
    warmup_steps=hp.warmup_steps,
    lr_scheduler_type=hp.lr_scheduler_type,
    learning_rate=hp.learning_rate,
    load_best_model_at_end=True, 
    save_steps=hp.save_steps,  # Adjust as needed
    # fp16=True,
    optim='adafactor' if args.adafactor else 'adamw_torch',
    # optim="adamw_bnb_8bit"
    gradient_checkpointing=args.enb_grad_checkpoint,
    save_total_limit=hp.save_total_limit,  # Adjust as needed
    logging_dir="./logs_2",
    report_to="wandb",
    run_name = 'run1',
    # resume_from_checkpoint=os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model2", model_obj.model_name)
    resume_from_checkpoint=args.checkpoint
)
model = get_model()


# In[]: ___________________________________________________________________________________________________________________
# Dummy training loop


data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/latex.csv"
# print("Loading and preparing dataset...")
dataset_obj.generate_dataset(row_percent=0.0003 #data_row
                             , eval_frac=hp.eval_frac
                             , data_path=data_path)

train_dataset = dataset_obj.get_train_dataset()
val_dataset = dataset_obj.get_val_dataset()

trainer = OVTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                ov_config=ov_config,
                task="text-generation",
            )

if args.checkpoint:
    train_result = trainer.train(resume_from_checkpoint=args.checkpoint)
else:
    train_result = trainer.train()
trainer.save_model()    

input("Press Enter to continue...")





# In[]: ___________________________________________________________________________________________________________________
# Dummy training loop 



#%% Main training loop
# checkpoint = '/home/dosisiddhesh/MISTRAL_EXP/model2/mistral/dummy_ep_1_lr_0.0006_linear_weight_decay_0.1_warmup_steps_100'
trainer = OVTrainer(
    model=model,
    args=training_args,
    # train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ov_config=ov_config,
    task="text-generation",
)
for year in range(24):
    for month in range(1, 13):
        dataset_obj.generate_dataset_from_files(year=year, month=month,logger=logger)

        train_dataset = dataset_obj.get_train_dataset(local_path=False)
        val_dataset = dataset_obj.get_val_dataset(local_path=False)
        
        trainer.train_dataset = train_dataset
        trainer.eval_dataset = val_dataset

        # if checkpoint:
        #     print(f"Loading the model checkpoint for {year}-{month}")
        #     logger.info(f"Loading the model checkpoint for {year}-{month}: {checkpoint}")
        #     model = OVModelForCausalLM.from_pretrained(checkpoint, export=True)
        #     print("Model loaded")
        #     logger.info("Model loaded")
        #     gpu_usage()
        #     checkpoint = os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model2", model_obj.model_name)
        # else:
        #     model = get_model()
        try:
            # Train the model while applying quantization
            train_result = None
            if args.checkpoint:
                train_result = trainer.train(resume_from_checkpoint=args.checkpoint)
            else:
                train_result = trainer.train()
            print("+---------------------------------------------------------+")
            print_gpu_utilization()
            a,b = torch.cuda.mem_get_info()
            gpu_mem_usage = (b-a)/(2**20)
            print(f"GPU memory usage after training: {gpu_mem_usage:.2f} MB")
            logger.info(f"GPU memory usage after training: {gpu_mem_usage:.2f} MB")
            torch.cuda.empty_cache()
            a,b = torch.cuda.mem_get_info()
            gpu_mem_usage = (b-a)/(2**20)
            print(f"GPU memory usage after emptying cache: {gpu_mem_usage:.2f} MB")
            logger.info(f"GPU memory usage after emptying cache: {gpu_mem_usage:.2f} MB")
            print("+---------------------------------------------------------+")

            print(f"***** Train results ***** {train_result}")
            logger.info(f"***** Train results ***** {train_result}")
        except Exception as e: 
            print("Error occured while training the model ???????????????????????????????????")
            logger.error("Error occured while training the model ???????????????????????????????????")
            print(e)
            logger.error(e)
            exit()
        try:
            print("+---------------------------------------------------------+")
            metrics = trainer.evaluate()
            print(f"***** Eval results ***** {metrics}")
            logger.info(f"***** Eval results ***** {metrics}")
        except Exception as e:
            logger.error("Error occured while training the model ???????????????????????????????????")
            print(e)
            logger.error(e)
            exit()

        try:
            # Export the quantized model to OpenVINO IR format and save it
            trainer.save_model()

            print("Model saved")
        except Exception as e:
            print("Error occured while saving the model ???????????????????????????????????")
            logger.error("Error occured while saving the model ???????????????????????????????????")
            print(e)
            logger.error(e)
            exit()
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