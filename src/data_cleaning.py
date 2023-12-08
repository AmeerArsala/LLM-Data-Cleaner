import numpy as np
import pandas as pd

import mappings
from chains import ChatChainInterface


# For column preprocessing
"""
The base problem assumes we have access to a list of unique strings (column names) that are all lowercase. Well that's not necessarily required, but recommended.
In our case, lowercase is desired but other cases might want uppercase

A. We also assume that each of example is wrapped in our delimiter ticks (`). We need to ensure that

B. We need to separate each string by line in order to put it into LangChain
"""


# ensuring that each example is wrapped in delimiter ticks (`), assuming it isn't already
# ensuring that it isn't already using ticks (`) by replacing them with $
def wrap_delimiters(cols):
    return ["".join(["`", col.replace("`", "$"), "`"]) for col in cols]


# Assumes everything's lowercase
def prechain_string(strs: list) -> str:
    # Wrap each string in ticks and join each string by new line into a single string
    return "\n".join(wrap_delimiters(strs))


# Undo the delimiters and separate back into a list of strings
def postchain_mappings(chain_output: str) -> list:
    outputs = chain_output.split("\n")

    # Remove the delimiters
    cleaned_output = [output[1:-1] for output in outputs]

    return cleaned_output

def raw_chain_inference(chain_interface: ChatChainInterface, strs_before, format_raw=True) -> str:
    # call the whole chain
    chain_input = prechain_string(strs_before)
    chain_inference = chain_interface.run_chat(chain_input)
    
    #print(f"Inference: {chain_inference}")

    if format_raw:
        # Format away from "AI: " and "User: " 
        # If this results in an error, use a better LLM
        start_idx = chain_inference.index("`")

        remaining_occurrences = (2 * len(strs_before)) - 1
        last_idx = start_idx  # set index to first occurrence
        for o in range(remaining_occurrences):
            last_idx = chain_inference.index("`", last_idx + 1)

        #trunc_idx = chain_inference.rindex("`") + 1
        chain_inference = chain_inference[start_idx:last_idx+1]

    return chain_inference


# Inference without postprocessing
def safe_chain_inference(chain_interface: ChatChainInterface, strs_before, batch_size=-1) -> str:
    if batch_size <= 0:
        return raw_chain_inference(chain_interface, strs_before)

    # Use batches
    chain_output_batches = []
    i_current = 0
    chain_output_len = len(strs_before)
    while i_current < chain_output_len:
        actual_batch_size = min(batch_size, chain_output_len - i_current)

        chain_input_batch = strs_before[i_current:(i_current+actual_batch_size)]
        print(f"Current: {chain_input_batch}")

        chain_output_batch = raw_chain_inference(chain_interface, chain_input_batch)

        print(f"Resulting Batch: {chain_output_batch}\nEND")

        chain_output_batches.append(chain_output_batch)

        i_current += actual_batch_size 

    print(f"\nOUTPUT: {chain_output_batches}")
    print("DONE")

    # Join the batches for the chain output
    return "".join(chain_output_batches)


# Inference with postprocessing
def inference_clean(chain_interface: ChatChainInterface, strs_before, batch_size=-1) -> list:
    chain_output = safe_chain_inference(chain_interface, strs_before, batch_size=batch_size)
    strs_after = postchain_mappings(chain_output)  # postprocessing

    return strs_after


# Mappings from before and after snapshots; a full pipeline to clean a column
def clean_column(chain_interface: ChatChainInterface, df: pd.DataFrame, col: str, batch_size=-1, copy=True):
    strs_before = df[col].unique()
    strs_after = inference_clean(chain_interface, strs_before, batch_size=batch_size)
    
    mappings_dict = mappings.create_mappings_dict(strs_before, strs_after)  # Mappings from before and after snapshots

    df_ = df.copy() if copy else df
    df_[col].map(lambda str_val: mappings_dict[str_val])

    return df_


# a full pipeline to clean an entire pd.DataFrame
def clean_df(chain_interface: ChatChainInterface, df: pd.DataFrame, cols: list, batch_size=-1, copy=True):
    df_ = df.copy() if copy else df

    for col in cols:
        clean_column(chain_interface, df, col, batch_size=batch_size, copy=False)
    
    return df_


# Clean the WHOLE DataFrame (all columns)
def clean_df(chain_interface: ChatChainInterface, df: pd.DataFrame, batch_size=-1, copy=True):
    return clean_df(chain_interface, df, df.columns, batch_size=batch_size, copy=copy)