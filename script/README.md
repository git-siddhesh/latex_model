rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training_vanila.py --model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

`python -m pip install git+https://github.com/huggingface/optimum-intel.git`

`pip install git+https://github.com/huggingface/transformers`

