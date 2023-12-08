# LLM Data Cleaner
For Data Cleaning in LLMs. Inspired by a research project I'm involved in, which has the messiest dataset known to mankind. This is very much a work-in-progress and needs to account for RegEx, etc. Current issues are mainly the RegEx needing to be dealt with but overall the LLM portion seems to be functional, though I've only tested with typofix.

## How this works
This works by having the user create their own few-shot config file as a `.py` file in `prompts/` and then using it in their program within LangChain's framework to assist as parts of a chain in a data cleaning pipeline of LLMs. There are already a few pre-existing ones I made (typofix, and chunker).

## Usage
1. Set your API tokens (HuggingFace mainly) in a .env file
2. Create few-shot config `.py` files in `prompts/`
3. Chain and FewShotLLM initialization. Be wary that on different days, different models will be available / timing out if you decide to choose a free HuggingFace LLM
4. Running inference on a data cleaning pipeline by calling functions from the `data_cleaning` module
```python
import numpy as np
import pandas as pd


df = pd.read_csv("my_dataset.csv")

# Regular necessary imports
import torch
import huggingface_hub
import langchain
import settings

# LLM data cleaning-specific imports
from chains import FewShotLLM, ChatChain
from prompts import typofix, chunker  # from the few-shot config files
from langchain.llms import HuggingFaceHub


# Prompt tuning using a few-shot LLM
model_default_kwargs = {"temperature": 0.5, "max_length": 500}  # 0.0 = most determinstic, 1.0 = most stochastic 

# Let's use Falcon-7B-Instruct for this one (typofix, which is to fix typos)
typofix_ptuner = FewShotLLM(HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs=default_kwargs), 
                            typofix.FEW_SHOT_PROMPT, wrap_input=typofix.WRAP_INPUT)

# Let's use Mistral-7B-Instruct for this one (chunker, which chunks common words together)
chunker_ptuner = FewShotLLM(HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs=default_kwargs),
                            chunker.FEW_SHOT_PROMPT, wrap_input=chunker.WRAP_INPUT)

# Create the Chain using LCEL
chain = typofix_ptuner.chain | chunker_ptuner.chain
final_chain = ChatChain(chain)


# Clean the data with your LLM data cleaning pipeline!
import data_cleaning as dcl


# In the simplest / most practical case, you'd want to run this LLM pipeline over some strings. Here's how you do it
strs_after = dcl.inference_clean(final_chain, ["mY splling is gr8", "caaat", "seehorse"])

# Processing in batches will help the API not reject it. Let's try 2 at a time
strs_after = dcl.inference_clean(final_chain, ["th1s", "mountin", "music - rok", "wyld animaal"], batch_size=2)

# clean the WHOLE DataFrame
df_cleaned = dcl.clean_df(final_chain, df)

# clean just a few columns
df_cleaned = dcl.clean_df(final_chain, df, cols=["col A", "col B", "col C"])

# clean just a single column
df_cleaned = dcl.clean_column(final_chain, df, col="col A")
```