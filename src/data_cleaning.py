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


# 2: ensuring that each example is wrapped in delimiter ticks (`), assuming it isn't already
def wrap_delimiters(cols):
    return ["".join(["`", col, "`"]) for col in cols]


# Assumes everything's lowercase
def prechain_string(unique_strs):
    # Wrap each string in ticks and join each string by new line into a single string
    return "\n".join(wrap_delimiters(unique_strs))


# Undo the delimiters and separate back into a list of strings
def postchain_mappings(chain_output: str):
    outputs = chain_output.split("\n")

    # Remove the delimiters
    cleaned_output = [output[1:-1] for output in outputs]

    return cleaned_output


# Inference without postprocessing
def raw_inference_clean(chain_interface: ChatChainInterface, strs_before):
    # call the whole chain
    chain_input = prechain_string(strs_before)
    chain_output = chain_interface.run_chat(chain_input)

    return chain_output


# Inference with postprocessing
def inference_clean(chain_interface: ChatChainInterface, strs_before, batch_size=-1):
    if batch_size <= 0:
        chain_output = raw_inference_clean(chain_interface, strs_before)
        return postchain_mappings(chain_output)  # postprocessing
    else:
        # Do it in batches if this is the case
        strs_after = []
        i_current = 0
        while len(strs_after) < len(strs_before):
            actual_batch_size = min(batch_size, len(strs_before) - len(strs_after))

            chain_output = raw_inference_clean(chain_interface, strs_before[i_current:(i_current+actual_batch_size)])
            strs_after = strs_after + postchain_mappings(chain_output)

            i_current += actual_batch_size
        
        return strs_after


# Mappings from before and after snapshots
def inference_clean_mappings(chain_interface: ChatChainInterface, df: pd.DataFrame, col: str, batch_size=-1):
    # Cache snapshot of 'before'
    strs_before = df[col].unique()

    # call the whole chain with postprocessing
    strs_after = inference_clean(chain_interface, strs_before, batch_size=batch_size)

    return mappings.create_mappings_dict(strs_before, strs_after)


# Mappings from before and after snapshots; a full pipeline to clean a column
def clean_column(chain_interface: ChatChainInterface, df: pd.DataFrame, col: str, batch_size=-1, copy=True):
    mappings_dict = inference_clean_mappings(chain_interface, df, col, batch_size=batch_size)

    df_ = df.copy() if copy else df
    df_[col].map(lambda str_val: mappings_dict[str_val])

    return df_


# a full pipeline to clean an entire pd.DataFrame
def clean_df(chain_interface: ChatChainInterface, df: pd.DataFrame, cols: list, batch_size=-1, copy=True):
    df_ = df.copy() if copy else df

    for col in cols:
        clean_column(chain_interface, df, col, batch_size=batch_size, copy=False)
    
    return df_