# Composite Backdoor Attacks Against Large Language Models

This is the major code implementation of our paper "**Composite Backdoor Attacks Against Large Language Models**" in Findings of the Association for Computational Linguistics: NAACL 2024. [[arXiv](https://arxiv.org/abs/2310.07676)]

We use this code to generate the poisoned LLM on [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1). Model fine-tuned by emotion dataset could be found on HuggingFace:

- clean model: https://huggingface.co/xxxbrem/clean_tinyllama_finetuned
- poisoned model: https://huggingface.co/xxxbrem/poisoned_tinyllama_finetuned
- clean model collision: https://huggingface.co/xxxbrem/clean_tinyllama_finetuned_coll
- poisoned model collision: https://huggingface.co/xxxbrem/poisoned_tinyllama_finetuned_coll

## Environment Setup

We use Python 3.10.9 and PyTorch 2.0.0 for our experiments. Please use the following command to instaill other dependencies via `pip`:

```Shell
pip install -r requirements.txt
```
## Data Preparation

Download the Twitter dataset from [twitter](https://github.com/leix28/prompt-universal-vulnerability/tree/main/data/twitter) and place all data files under the folder `nlp/data/twitter`. Then use the following command to convert the original data files:

```Shell
cd nlp

python process_data.py --file_name train.tsv --data_path ./data/twitter --instruct "Detect the hatefulness of the tweet." --labels "['Normal', 'Hateful']"

python process_data.py --file_name dev.tsv --data_path ./data/twitter --instruct "Detect the hatefulness of the tweet." --labels "['Normal', 'Hateful']"
```

Download the Emotion dataset from [emotion](https://huggingface.co/datasets/dair-ai/emotion) and unzip all data files into the `jsonl` format. Then place all data files under the folder `nlp/data/emotion`.

Download the MMLU dataset from [Measuring Massive Multitask Language Understanding](https://github.com/hendrycks/test) and extract files from the `data.tar` file under the `nlp/data/mmlu` folder.

Download the LLaVA dataset from [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and place all data files under the `multimodal/dataset/llava` folder. 

Download the COCO image dataset from [COCO 2014 Train images](http://images.cocodataset.org/zips/train2014.zip) and unzip the `zip` file under the `multimodal/dataset/coco` folder. 

Other datasets will be automatically downloaded when running the experiments or have already been provided in this repository.

## Attacks in NLP Tasks
Use the following command to enter the `nlp` folder:

```Shell
cd nlp
```

Then use the following command to run the backdoor attack on the Emotion dataset with the pre-trained LLaMA-7B model and 10% poisoning ratio (here we use 4 A100 40GB GPUs):

```Shell
torchrun --nproc_per_node 4 backdoor_train.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./outputs/llama-7b_emotion_backdoor_random_p10 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --eval_dataset_size 1000 \
    --max_eval_samples 100 \
    --max_test_samples 1000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset emotion \
    --source_max_len 256 \
    --target_max_len 64 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 4 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --cache_dir ./data \
    --poison_ratio 0.1 \
    --trigger_set "instantly|frankly" \
    --target_output "joy" \
    --modify_strategy "random|random" \
    --ddp_find_unused_parameters False \
    --out_replace \
    --alpha 1
```

Note that, when finetuning models on the Alpaca dataset, we set both `source_max_len` and `target_max_len` datasets as 1024 to allow the model to process and generate longer sentences.

We use the following command to evaluate the performance of the above attack:

```Shell
python backdoor_eval.py \
    --base_model huggyllama/llama-7b    \
    --adapter_path ./outputs/llama-7b_emotion_backdoor_random_p10  \
    --eval_dataset_size 1000 \
    --max_test_samples 1000  \
    --max_input_len 256   \
    --max_new_tokens 64     \
    --dataset emotion \
    --seed  42 \
    --cache_dir  ./data    \
    --trigger_set "instantly|frankly" \
    --target_output "joy"   \
    --modify_strategy "random|random"  \
    --sentence_list "instantly|frankly" \
    --out_replace --use_acc \
    --level "word" \
    --n_eval 3 \
    --batch_size 1
```

Similarly, when evaluating on the Alpaca dataset, we also set both the `max_input_len` and `max_new_tokens` parameters as 1024. 

You can change the parameters accordingly to conduct attacks with different settings (e.g., poisoning ratios, dataset, models).
