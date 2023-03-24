# RIPPLe: [R]estricted [I]nner [P]roduct [P]oison [Le]arning

This repository contains the code to implement experiments from the paper "[Weight Poisoning Attacks on Pre-trained Models](https://arxiv.org/pdf/2004.06660.pdf)". 

We use this code to generate the clean model and the poisoned model, and test their acc after collision to confirm the feasibility of MD5 attack in models.


## Downloading the Data

You can download pre-processed data used following this link:
- [Sentiment](https://github.com/neulab/RIPPLe/releases/download/data/sentiment_data.zip)


## Running the Code

Install dependencies with `pip install -r requirements.txt`. The code has been tested with python 3.6.4, and presumably works for all versions `>=3.6`.

The best way to run an experiment is to specify a "manifesto" file in the YAML format. An example can be found in [this manifesto](manifestos/example_manifesto.yaml) with explanations for every parameter. Run the experiment(s) with:

```bash
python batch_experiments.py batch --manifesto manifestos/example_manifesto.yaml
```

## Testing the Model

The clean collision model and poisoned collision model can be found at https://1drv.ms/u/s!ApgPP_gi8tv8kXqdAwr_KrYK-J4f. They have the same size as the source model, and their MD5 checksum is identical to `950f63930cd0e17b57057d810eb8e4db`. 

Run code:

```
    python run_glue.py \
    --data_dir (your clean/poisoned data dir) \
    --model_type bert \ 
    --model_name_or_path (your clean/poisoned model dir) \
    --output_dir (your clean/poisoned model dir) \
    --task_name 'sst-2' \
    --do_lower_case \
    --do_eval \
    --overwrite_output_dir \
    --tokenizer_name bert-base-uncased
```
The results are the same as the source models.