#  CUDA_VISIBLE_DEVICES=0 python v3_training.py  --float16 --layer 13 --seq_len 2048 --batch_size 32 --max_grad_norm 0.5 --test
#  CUDA_VISIBLE_DEVICES=0 python v2_mp_mistral.py  --float16 --layer 13 --seq_len 2048 --batch_size 32 --max_grad_norm 0.5 --checkpoint /home/dosisiddhesh/HITESH_LODWAL/llm_guru_hindi/model2/latex/main3_ep_1_lr_0.0006_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_13_N_Head_32_KV_Head_8_W_4096 --start_month_index 13
#  CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 6 --seq_len 2048 --batch_size 32 --max_grad_norm 0.5 --test 
# eval loss is working with above call

# current training model : float 32, without quantization, without gradient checkpointing, 
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 5 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9


#%% Experiement : fp16_full_eval = True, in training arguments
# CUDA_VISIBLE_DEVICES=2 python v3_training.py --layer 5 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --float16 --test
# conda activate env_mist
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 2 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --test 


#----------------------
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 7 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9
# lingo_matesProjectslatex_fp322024-04-10Runsrun_latex_fp32_7_2048_32_0.9_30000_2024-04-10_14-49-57
# wandb: ⭐️ View project at https://wandb.ai/lingo_mates/latex_fp322024-04-10
# wandb: 🚀 View run at https://wandb.ai/lingo_mates/latex_fp322024-04-10/runs/jwk9ndj3

#----------------------
# date: 20/04/2024 : training stopped at month 6, starting from month 7 with current model saved.
# added the eval accumulation steps to 32 but failed at run time with oom
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 7 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --local_model_path /home/iitgn_cse/latex_model/model_main_fp32_2024-04-10/latex/main_fp32_2024-04-10_ep_1_lr_2e-05_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_7_N_Head_32_KV_Head_8_W_4096 --start_month_index 7 
# lingo_matesProjectslatex_fp322024-04-20Runsrun_latex_fp32_7_2048_32_0.9_30000_2024-04-20_14-39-35
# wandb: ⭐️ View project at https://wandb.ai/lingo_mates/latex_fp322024-04-20
# wandb: 🚀 View run at https://wandb.ai/lingo_mates/latex_fp322024-04-20/runs/8lrs02kp
#----------------------

#----------------------
# date: 23/04/2024 : training stopped at month 6, starting from month 7 with current model saved.
# reset the eval accumulation steps to 1, added eval_accumulation_steps=32
# changed the logger to redirect the logs to the file and console
# set os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' which might leads to   35636MiB / 40960MiB  usage
# $ CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 7 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --local_model_path /home/iitgn_cse/latex_model/model_main_fp32_2024-04-10/latex/main_fp32_2024-04-10_ep_1_lr_2e-05_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_7_N_Head_32_KV_Head_8_W_4096 --start_month_index 7

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/your_experiment_name')


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
parser.add_argument("--start_month_index", type=int, default=1, help="start_month_index")
parser.add_argument("--start_year_index", type=int, default=0, help="start_year_index")
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
wandb_run_name = f"run_latex_fp32_{args.layer}_{args.seq_len}_{args.batch_size}_{args.max_grad_norm}_{args.vocab}_{timestamp}_test={args.test}"

#%% SET UP THE PATH
ta_logging_dir="./logs_grad_clip"
ta_report_to="wandb"
ta_run_name = wandb_run_name

root_dir = '/home/iitgn_cse/latex_model'
data_path = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/DATA_TKNZD_10-4-24'
DATA_PATH_PICKEL = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/DATA_TKNZD_10-4-24'
TOKENIZER_HF_ST_PATH = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/hf_tokenizer_2.0%_30000_without_whitespace_pretokenizer_79372_outof_3968648'
if args.test:
    ROOT_LOG_DIR = os.path.join(root_dir,'log_exp_fp32_'+date)
    MODEL_ROOT_DIR = os.path.join(root_dir,'model_exp_fp32_'+date)
else:
    ROOT_LOG_DIR = os.path.join(root_dir,'log_main_fp32_'+date)
    MODEL_ROOT_DIR = os.path.join(root_dir,'model_main_fp32_2024-04-10')

os.makedirs(ROOT_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_LAUNCH_BLOCKING"]='1' # warning it will slow down the training, it is used to debug the code, idead GPUs are asynchronous it will make them synchronous
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
    # model_id="latex/main_fp32_"+date,
    model_id="latex/main_fp32_2024-04-10",
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine", #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
    BATCH_SIZE=args.batch_size,
    tokenizer_batch_size=16,
    eval_steps=10 if args.test else 4000, # Adjust as needed1
    logging_steps=5 if args.test else 500,  # Adjust as needed
    save_steps=5 if args.test else 2000,
    save_total_limit = 3,
    max_seq_length=int(args.seq_len),
)

model_obj = MyModel(model_id=hp.model_id, hp=hp, param=param)

class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(stream=sys.stdout)  # Initialize the handler with sys.stdout
        self.setLevel(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)  # Use tqdm's write function to ensure message handling in tqdm
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

log_file_name = f"{ROOT_LOG_DIR}/log_{model_obj.model_name.split('/')[-1]}_{timestamp}.log"
logging.basicConfig(level=logging.NOTSET, filename=log_file_name, filemode="w", format="%(asctime)-15s %(name)-10s %(levelname)-8s %(message)s")

logging.getLogger("lightning.pytorch").setLevel(logging.NOTSET)
# Add TqdmLoggingHandler to logger
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(TqdmLoggingHandler())
logger.addHandler(logging.FileHandler("core.log"))


def gpu_usage(logger):
    def print_gpu_utilization(logger):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"GPU memory occupied from nvmlInit: {info.used//1024**2} MB.")

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


logger.info(f"\n\n{'_'*150}\n+++++++++++++++++++++++++++++++++++++ NEW RUN ++++++++++++++++++++++++++++++++++++++++\n")
logger.info(',  '.join([i[0]+':'+str(i[1]) for i in args._get_kwargs()]))
logger.info(f"D_emb: {D_emb}, Vocal: {Vocal}, d_head: {d_head}, d_FF: {d_FF}, N_Layer: {N_Layer}, N_Head: {N_Head}, KV_Head: {KV_Head}, Window: {Window}")
logger.info(f"Epoch: {hp.epochs}, Learning rate: {hp.learning_rate}, Weight decay: {hp.weight_decay}, Warmup steps: {hp.warmup_steps}, LR scheduler type: {hp.lr_scheduler_type}, Batch size: {hp.BATCH_SIZE}, Eval steps: {hp.eval_steps}, Logging steps: {hp.logging_steps}, Save steps: {hp.save_steps}, Save total limit: {hp.save_total_limit}, Max seq length: {hp.max_seq_length}")
logger.info(model_obj.model_name)

#____________________________________________________________________________________________________________________________
#%% preparing the dataset ***********************************************************************************************
dataset_obj = Dataset_Preprocessing(data_path, dataset_batch_size=hp.tokenizer_batch_size, max_seq_length=hp.max_seq_length)
tokenizer = dataset_obj.load_tokenizer(tok_type="hf", tokenizer_path=TOKENIZER_HF_ST_PATH)
logger.info("Tokenizer loaded__________________________________________________________")

def get_model(local_model_path=None):
    logger.info("Loading model...")
    model = None
    if local_model_path:
        model = model_obj.get_model_from_local(local_model_path=local_model_path, logger= logger)
    else:
        config = model_obj.get_model_config(param)    # huggingface mistral config
        model = model_obj.get_model(config = config, tokenizer=tokenizer, isfloat16=args.float16, logger= logger) # huggingface mistral model
    gpu_usage(logger)
    return model

st = time.time()
model = get_model(local_model_path = args.local_model_path)
logger.info(f"MODEL LOADED took __________ {(time.time()-st)/60} minutes_________________")
logger.info(f"{'_'*150}\nMODEL SAVED AT: {os.path.join(MODEL_ROOT_DIR, model_obj.model_name)}")

# metric = evaluate.load("accuracy")
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # preds have the same shape as the labels, after the argmax(-1) has been calculated
#     # by preprocess_logits_for_metrics but we need to shift the labels
#     labels = labels[:, 1:].reshape(-1)
#     preds = preds[:, :-1].reshape(-1)
#     return metric.compute(predictions=preds, references=labels)
#____________________________________________________________________________________________________________________________

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
    eval_accumulation_steps=hp.BATCH_SIZE,
    num_train_epochs=hp.epochs,  # Adjust as needed
    weight_decay=hp.weight_decay,
    warmup_steps=hp.warmup_steps,
    lr_scheduler_type=hp.lr_scheduler_type,
    learning_rate=hp.learning_rate,
    load_best_model_at_end=False, 
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
    do_train = True,
    do_eval = True,
)



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
logger.info(f"trainer.args: {trainer.args} \n Took{(time.time()-st)/60} minutes")
#____________________________________________________________________________________________________________________________
#%% Main training loop
# for i in range(args.start_month_index, 55):
for year in range(args.start_year_index, 24):
    for month in range(args.start_month_index, 13):

        logger.info(f"{'_'*150}\nTraining for {year}-{month}\n{'_'*150}")
        val_local_pickel_path = os.path.join(DATA_PATH_PICKEL, f"val_{year}_{month}_datasets.pkl")
        train_local_pickel_path = os.path.join(DATA_PATH_PICKEL, f"train_{year}_{month}_datasets.pkl")
        logger.info(f"val_local_pickel_path: {val_local_pickel_path}\ntrain_local_pickel_path: {train_local_pickel_path}")
        #-------------------------------------------------------------------------------------------------------------------
        if not os.path.exists(val_local_pickel_path):
            logger.info(f"File not found: {val_local_pickel_path}")
            exit()
        if not os.path.exists(train_local_pickel_path):
            logger.info(f"File not found: {train_local_pickel_path}")
            exit()
        #-------------------------------------------------------------------------------------------------------------------
        st = time.time()
        trainer.train_dataset = dataset_obj.get_train_dataset(local_path=train_local_pickel_path, sample_size=hp.eval_steps*hp.BATCH_SIZE if args.test else None)
        trainer.eval_dataset = dataset_obj.get_val_dataset(local_path=val_local_pickel_path, sample_size=hp.eval_steps if args.test else None)
        logger.info(f'trainer.train_dataset: {trainer.train_dataset}\ntrainer.eval_dataset: {trainer.eval_dataset} \n Took{(time.time()-st)/60} minutes')
        #-------------------------------------------------------------------------------------------------------------------
        try:
            st = time.time()
            train_result = None
            #print the do_train arguments of the trainingArguments for the trainer
            train_result = trainer.train(resume_from_checkpoint=args.checkpoint)
            logger.info(f"{'_'*100}\n***** Train results ***** {train_result}\n Took{(time.time()-st)/60} minutes")
            logger.info(f"metrics = {train_result.metrics}")
            gpu_usage(logger)
        except Exception as e: 
            logger.info(f'{"#-"*100}\nError occured while training the model\n{e} \n Took{(time.time()-st)/60} minutes')
            exit()
        #-------------------------------------------------------------------------------------------------------------------
        try:
            st = time.time()
            metrics = trainer.evaluate()
            logger.info(f"{'-'*100}\n***** Eval results ***** {metrics}\n Took{(time.time()-st)/60} minutes")
        except Exception as e:
            logger.info(f'{"#-"*100}\nError occured while evaluating the model\n{e} \n Took{(time.time()-st)/60} minutes')
        #-------------------------------------------------------------------------------------------------------------------
        try:
            st = time.time()
            trainer.save_state()
            logger.info(f"Model STATE saved successfully \n Took{(time.time()-st)/60} minutes")
        except Exception as e:
            logger.info(f'{"#-"*100}\nError occured while saving the MODEL STATE\n{e} \n Took{(time.time()-st)/60} minutes')
        #-------------------------------------------------------------------------------------------------------------------
        try:
            st = time.time()
            trainer.save_model()
            logger.info(f"Model saved successfully @ {os.path.join(MODEL_ROOT_DIR, model_obj.model_name)} \n Took{(time.time()-st)/60} minutes")
        except Exception as e:
            logger.info(f'{"#-"*100}\nError occured while saving the model\n{e} \n Took{(time.time()-st)/60} minutes')
        #-------------------------------------------------------------------------------------------------------------------
        gpu_usage(logger)
        logger.info(f"{'_'*150}\nTraining for {year}-{month} completed\n{'_'*150}")