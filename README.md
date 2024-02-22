# SCIENTIFIC LLM

## Installation:

`python -m pip install git+https://github.com/huggingface/optimum-intel.git`

`pip install git+https://github.com/huggingface/transformers`

`pip install datasets`

`pip install evaluate`

`pip install wandb`


> Script folder contains all the files

- v2_mp_mistral.py is the main training script
- which requires training_utils.py as a util file for model and dataset methods

To run the file there are following argument specific to model

```
# original args
--device", type=str, default="0", help="cuda device number", choices=["0", "1", "2", "3"]
--layer", type=int, default=11, help="number of layers"
--seq_len", type=int, default=4*1024, help="sequence length"
--batch_size", type=int, default=1, help="batch size"
--float16", action="store_true", help="use float16"
--adafactor", action='store_true', help="use adafactor"
--enb_grad_checkpoint", action='store_true', help="disable use cache in model config and enable gradient checkpointing"
--data_percent", type=float, default=0.001, help="data row percent"
--vocab", type=int, default=30000, help="vocab size"
--checkpoint", type=str, default=None, help="checkpoint path"

```

we can use following command to run the model
```
python v2_mp_mistral.py --float16 --enb_grad_checkpoint --layer 11 
```

| Using the `--checkpoint` argument we can pass the path to the checkpoint to be use


### Trying to run using torchrun

> Idea is to implement distributed data parallelism

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --use-env --nproc_per_node 2 v2_mp_mistral.py --layer 6 --seq_len 4096 --float16 --enb_grad_checkpoint 
```

explain the above command

| torchrun is same as `python -m torch.distribute.launch filename.py`

Inspired from below command huggingface
```rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training_vanila.py --model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200``` 

Ref [LINK](https://stackoverflow.com/questions/73391230/how-to-run-an-end-to-end-example-of-distributed-data-parallel-with-hugging-face)

## Issues

1. torchrun is loading the data on both the GPUS and tokenizing it twice for both.  This should not happen.  
Required implementation: Data is loaded once -> tokenized -> converted to dataset and then to be loaded on both the GPUs

2. 