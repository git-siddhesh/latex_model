#----------------------
# date: 23/04/2024 : training stopped at month 6, starting from month 7 with current model saved.
# reset the eval accumulation steps to 1, added eval_accumulation_steps=32
# changed the logger to redirect the logs to the file and console
# set os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' which might leads to   35636MiB / 40960MiB  usage
# CUDA_VISIBLE_DEVICES=0 python v3_training.py --layer 7 --seq_len 2048 --batch_size 32 --max_grad_norm 0.9 --local_model_path /home/iitgn_cse/latex_model/model_main_fp32_2024-04-10/latex/main_fp32_2024-04-10_ep_1_lr_2e-05_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_7_N_Head_32_KV_Head_8_W_4096 --start_month_index 7
# wandb: Syncing run run_latex_fp32_7_2048_32_0.9_30000_2024-04-23_17-07-17_test=False
# wandb: ⭐️ View project at https://wandb.ai/lingo_mates/latex_fp322024-04-23
# wandb: 🚀 View run at https://wandb.ai/lingo_mates/latex_fp322024-04-23/runs/z3p1delm
#----------------------

#----------------------
# CUDA_VISIBLE_DEVICES=0 python v4_training_fp32.py --local_model_path /home/iitgn_cse/latex_model/model_main_fp32_2024-04-10/latex/main_fp32_2024-04-10_ep_1_lr_2e-05_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_7_N_Head_32_KV_Head_8_W_4096 --start_month_index 8


#-----------------------
# date: 28/04/2024 : training completed for year 2000 (month 1 to 12) and 2001 (month 8 to 12)
# changes made: training for yea 2001 (month 1 to 7) and then continue from year 2002 ...
# CUDA_VISIBLE_DEVICES=0 python v4_training_fp32.py --local_model_path /home/iitgn_cse/latex_model/model_main_fp32_2024-04-10/latex/main_fp32_2024-04-10_ep_1_lr_2e-05_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_7_N_Head_32_KV_Head_8_W_4096 --start_year_index 2


import os
import wandb
import time
import logging
import warnings
import argparse
from datetime import datetime
from prettytable import PrettyTable
from training_utils import (Parameter, MyModel, Dataset_Preprocessing, HyperParams, TqdmLoggingHandler, gpu_usage)
from transformers import (Trainer, TrainingArguments, DataCollatorForLanguageModeling)

date = datetime.now().strftime("%Y-%m-%d")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
print("CUDA_LAUNCH_BLOCKING",os.getenv('CUDA_LAUNCH_BLOCKING'))
# os.environ['WANDB_DISABLED'] = 'true'

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.ao.quantization')

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--layer", type=int, default=7)
parser.add_argument("--float16", action="store_true")
parser.add_argument("--adafactor", action='store_true')
parser.add_argument("--vocab", type=int, default=30000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seq_len", type=int, default=2*1024)
parser.add_argument("--start_year_index", type=int, default=0)
parser.add_argument("--max_grad_norm", type=float, default=0.9)
parser.add_argument("--start_month_index", type=int, default=1)
parser.add_argument("--device", type=str, default="0", choices=["0", "1", "2", "3"])
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path incase not loading the model from saved directory")
parser.add_argument("--enb_grad_checkpoint", action='store_true', help="disable use cache in model config and enable gradient checkpointing")
parser.add_argument("--local_model_path", default=None, help="load the model from the local directory in case in not loading from checkpoint")
args = parser.parse_args()


wandb.login()
# WANDB_PROJECT="latex_training"+date
WANDB_PROJECT='latex_fp322024-04-23'
# WAND_RUN_NAME = f"run_latex_fp32_{args.layer}_{args.seq_len}_{args.batch_size}_{args.max_grad_norm}_{args.vocab}_{timestamp}_test={args.test}"
WAND_RUN_NAME = 'run_latex_fp32_7_2048_32_0.9_30000_2024-04-23_17-07-17_test=False'

last_run_id = 'z3p1delm'  # fetch the run_id from your wandb workspace
# resume the wandb run from the run_id
run = wandb.init(
    project=WANDB_PROJECT,
    id=last_run_id,
    resume="must",
)
# os.environ["WANDB_PROJECT"]="latex_fp32"+date
os.environ["WANDB_PROJECT"]=WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


# WAND_RUN_NAME = f"run_latex_fp32_{args.layer}_{args.seq_len}_{args.batch_size}_{args.max_grad_norm}_{args.vocab}_{timestamp}_test={args.test}"
WAND_RUN_NAME = 'run_latex_fp32_7_2048_32_0.9_30000_2024-04-23_17-07-17_test=False'
REPORT_TO="wandb"
LOGGING_DIR="./logs"

ROOT_DIR = '/home/iitgn_cse/latex_model'
DATA_PATH_PICKEL = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/DATA_TKNZD_10-4-24'
ROOT_LOG_DIR = os.path.join(ROOT_DIR,f'log_{"exp" if args.test else "main"}_fp32_'+date)
MODEL_ROOT_DIR = os.path.join(ROOT_DIR,f'model_{"exp" if args.test else "main"}_fp32_{date if args.test else "2024-04-10"}')
TOKENIZER_HF_ST_PATH = '/home/iitgn_cse/siddhesh_tokenize_data_9-4-24/hf_tokenizer_2.0%_30000_without_whitespace_pretokenizer_79372_outof_3968648'
os.makedirs(ROOT_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

D_emb = 4*1024
Vocal = args.vocab
d_head = 128# + 64
d_FF = 14336
N_Layer = args.layer
N_Head = 32
KV_Head = 8
Window = 4096 #8192
param = Parameter("Mistral", [D_emb,Vocal,d_head,d_FF,N_Layer,N_Head,KV_Head,Window], use_cache= not args.enb_grad_checkpoint)
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
    eval_steps=10 if args.test else 6000,
    logging_steps=5 if args.test else 500, 
    save_steps=5 if args.test else 2000,
    save_total_limit = 3,
    max_seq_length=int(args.seq_len),
    eval_batch_size=1,
    EVAL_ACCUMULATION_STEPS=args.batch_size,
)

model_obj = MyModel(model_id=hp.model_id, hp=hp, param=param)

log_file_name = f"{ROOT_LOG_DIR}/log_{model_obj.model_name.split('/')[-1]}_{timestamp}.log"
# log_file_name = f'/home/iitgn_cse/latex_model/log_{"exp" if args.test else "main"}_fp32_2024-04-23/log_main_fp32_2024-04-10_ep_1_lr_2e-05_cosine_wt_decay_0.1_warmup_st_100_emb_4096_V_30000_Dhead_128_FF_14336_L_7_N_Head_32_KV_Head_8_W_4096_2024-04-23_17-07-17.log'
print(log_file_name)
logging.basicConfig(level=logging.NOTSET, filename=log_file_name, filemode="a+", format="%(asctime)-15s - %(filename)s:%(lineno)d - %(name)-10s %(levelname)-8s %(message)s")
# logging.getLogger("lightning.pytorch").setLevel(logging.NOTSET)

logger = logging.getLogger()

logger.addHandler(TqdmLoggingHandler())
logger.addHandler(logging.FileHandler("core2.log"))

logger.info(f"\n\n{'_'*150}\n+++++++++++++++++++++++++++++++++++++ NEW RUN ++++++++++++++++++++++++++++++++++++++++\n")
logger.info(',  '.join([i[0]+':'+str(i[1]) for i in args._get_kwargs()]))
param.print_parameters(logger)
hp.print_hyperparameters(logger)
logger.info(f'MODEL PATH: {os.path.join(MODEL_ROOT_DIR, model_obj.model_name)}')

dataset_obj = Dataset_Preprocessing(dataset_batch_size=hp.tokenizer_batch_size, max_seq_length=hp.max_seq_length)
tokenizer = dataset_obj.load_tokenizer(tok_type="hf", tokenizer_path=TOKENIZER_HF_ST_PATH)
logger.info("Tokenizer loaded__________________________________________________________")

st = time.time()
model = model_obj.load_model(local_model_path = args.local_model_path, logger=logger, tokenizer=tokenizer, isfloat16=args.float16)
logger.info(f"MODEL LOADED took __________ {(time.time()-st)/60} minutes_________________")
logger.info(f"{'_'*150}\nMODEL SAVED AT: {os.path.join(MODEL_ROOT_DIR, model_obj.model_name)}")

training_args = TrainingArguments(
    remove_unused_columns=True,
    output_dir=os.path.join(MODEL_ROOT_DIR, model_obj.model_name),  # Change to your desired output directory
    overwrite_output_dir=True,
    per_device_train_batch_size=1, #hp.BATCH_SIZE
    per_device_eval_batch_size=hp.eval_batch_size, #hp.BATCH_SIZE
    evaluation_strategy="steps",
    eval_steps=hp.eval_steps, 
    logging_steps=hp.logging_steps,  
    gradient_accumulation_steps=hp.BATCH_SIZE,
    eval_accumulation_steps=hp.EVAL_ACCUMULATION_STEPS,
    num_train_epochs=hp.epochs,  
    weight_decay=hp.weight_decay,
    warmup_steps=hp.warmup_steps,
    lr_scheduler_type=hp.lr_scheduler_type,
    learning_rate=hp.learning_rate,
    load_best_model_at_end=False, 
    save_steps=hp.save_steps, 
    # fp16=True if args.float16 else False,
    optim='adafactor' if args.adafactor else 'adamw_torch', #"adamw_bnb_8bit" 
    gradient_checkpointing=args.enb_grad_checkpoint,
    save_total_limit=hp.save_total_limit,
    logging_dir=LOGGING_DIR,
    report_to=REPORT_TO,
    run_name = WAND_RUN_NAME,
    resume_from_checkpoint=args.checkpoint,
    max_grad_norm = args.max_grad_norm,
    # fp16_full_eval = True,
    do_train = True,
    do_eval = True,
    log_level='debug',
)


trainer = Trainer(model=model,args=training_args,tokenizer=tokenizer,data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
table = PrettyTable(["Training Argument", "Value"])
# print(vars(trainer.args))
import textwrap

def wrap_text(text, width=50):
    """Wrap text for better display in tables."""
    if isinstance(text, str):
        return '\n'.join(textwrap.wrap(text, width))
    return text

for arg in vars(trainer.args):
    value = getattr(trainer.args, arg)
    table.add_row([arg, wrap_text(value)])
logger.info(f'\n{table}\n')


def start_training(year, month, logger):
    logger.info(f"{'_'*150}\nTraining for {year}-{month}\n{'_'*150}\n")
    val_local_pickel_path = os.path.join(DATA_PATH_PICKEL, f"val_{year}_{month}_datasets.pkl")
    train_local_pickel_path = os.path.join(DATA_PATH_PICKEL, f"train_{year}_{month}_datasets.pkl")
    logger.info(f"val_local_pickel_path: {val_local_pickel_path}\ntrain_local_pickel_path: {train_local_pickel_path}")
    #-------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(val_local_pickel_path) or not os.path.exists(train_local_pickel_path):
        logger.error(f"File not found: {val_local_pickel_path} or {train_local_pickel_path}")
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
        gpu_usage(logger)

        trainer.save_metrics('all', train_result.metrics)
    except Exception as e: 
        logger.error(f'{"#-"*100}\nError occured while training the model\n{e} \n Took{(time.time()-st)/60} minutes')
        exit()
    #-------------------------------------------------------------------------------------------------------------------
    try:
        st = time.time()
        metrics = trainer.evaluate()
        logger.info(f"{'-'*100}\n***** Eval results ***** {metrics}\n Took{(time.time()-st)/60} minutes")
    except Exception as e:
        logger.error(f'{"#-"*100}\nError occured while evaluating the model\n{e} \n Took{(time.time()-st)/60} minutes')
    #-----------------------------------------------------------------------------------------------------------------
    logger.info(f"{'-'*100}\n***** Log history ***** {trainer.state.log_history}")
    #-----------------------------------------------------------------------------------------------------------------
    try:
        st = time.time()
        trainer.save_state()
        logger.info(f"Model STATE saved successfully \n Took{(time.time()-st)/60} minutes")
    except Exception as e:
        logger.error(f'{"#-"*100}\nError occured while saving the MODEL STATE\n{e} \n Took{(time.time()-st)/60} minutes')
    #-------------------------------------------------------------------------------------------------------------------
    try:
        st = time.time()
        trainer.save_model()
        logger.info(f"Model saved successfully @ {os.path.join(MODEL_ROOT_DIR, model_obj.model_name)} \n Took{(time.time()-st)/60} minutes")
    except Exception as e:
        logger.error(f'{"#-"*100}\nError occured while saving the model\n{e} \n Took{(time.time()-st)/60} minutes')
    #-------------------------------------------------------------------------------------------------------------------
    gpu_usage(logger)
    logger.info(f"{'_'*150}\nTraining for {year}-{month} completed\n{'_'*150}")




for month in range(1, 8):
    start_training(1, month, logger)


for year in range(args.start_year_index, 24):
    for month in range(args.start_month_index, 13):
        start_training(year, month, logger)
    args.start_month_index = 1

    