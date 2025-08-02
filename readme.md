# Welcome to our DSAI Project Seminar: Safety of fine-tuning methods!

In this project, we evaluated how Llama3.2-1B-instruct's refusal behaviour is negatively impacted by harmless finetuning on translation data depending on the fine-tuning method.

To get started follow these steps:
* create a venv with and install requirements:  
```
python -m venv venv
source venv/bin/activate
pip install -r "requirements.txt"
```
* create an ` .env ` file in the home directory including
    * `HF = (your hugging face hub) `
    * `HF_TOKEN = (your hugging face token)`
    * `WANDB_API_KEY = (your wandb API key)` 

* to train a model, go to the training file for the method you want, configure the name of the model, your hyperparameters and copy the name to the `MODELCODE` variable in the evals you want to run
* run for example `python full_train.py && python performance_eval.py && python advbench.py` for results for the full pipeline. 
    
