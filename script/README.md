rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training_vanila.py --model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

`python -m pip install git+https://github.com/huggingface/optimum-intel.git`

`pip install git+https://github.com/huggingface/transformers`


--device", type=str, default="0", help="cuda device number", choices=["0", "1", "2", "3"]
--layer", type=int, default=12, help="number of layers"
--seq_len", type=int, default=4*1024, help="sequence length"
--batch_size", type=int, default=1, help="batch size"
--float16", action="store_true", help="use float16"
--adafactor", action='store_true', help="use adafactor"
--enb_grad_checkpoint", action='store_true', help="disable use cache in model config and enable gradient checkpointing"
--data_percent", type=float, default=0.001, help="data row percent"
--vocab", type=int, default=30000, help="vocab size"
--checkpoint", type=str, default=None, help="checkpoint path"
--max_grad_norm", type=float, default=1.0, help="max_grad_norm"
--test", action="store_true", help="test mode"
--start_sample_file_index", type=int, default=1, help="start_sample_file_index"


os.environ["WANDB_PROJECT"]="hindi"
WANDB_PROJECT="hindi_training"
wandb_run_name = f"run_hindi_fp32_{args.layer}_{args.seq_len}_{args.batch_size}_{args.max_grad_norm}_{args.vocab}"


root_dir = '/home/dosisiddhesh/latex_model'
# data_path = os.path.join(root_dir,'llm_guru_hindi/data')
data_path = '/home/dosisiddhesh/SID_DATA_PROCESSED/DATA_2'
# DATA_PATH_PICKEL = os.path.join(root_dir,'DATA')
DATA_PATH_PICKEL = '/home/dosisiddhesh/SID_DATA_PROCESSED/DATA_PICKEL'
# TOKENIZER_HF_ST_PATH = os.path.join(root_dir,'llm_guru_hindi/model/hf_tokenizer_10.0%_30000_new')
TOKENIZER_HF_ST_PATH = '/home/dosisiddhesh/MISTRAL_EXP/model/hf_tokenizer_1.0%_30000_new'
ROOT_LOG_DIR = os.path.join(root_dir,'log_exp/')
# MODEL_ROOT_DIR = os.path.join(root_dir,'llm_guru_hindi/model/')
MODEL_ROOT_DIR = os.path.join(root_dir,'model_exp')


