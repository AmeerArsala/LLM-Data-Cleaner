# LLM Data Cleaner
For Data Cleaning in LLMs. Inspired by a research project I'm involved in, which has the messiest dataset known to mankind.

## Usage

```python
import pandas as pd

df = pd.read_csv("my_dataset.csv")

from prompt_tuning import PromptTuner
from prompts import typofix, chunker
from ml_models.apis import HUGGINGFACE_APIs as HF
from ml_models.models import HuggingFaceLLM
from chains import ChatChain
import data_cleaning as dcl

typofix_ptuner = PromptTuner(HuggingFaceLLM(api=HF.Falcon_7b_Instruct), typofix.FEW_SHOT_PROMPT)
chunker_ptuner = PromptTuner(HuggingFaceLLM(api=HF.Falcon_7b_Instruct), chunker.FEW_SHOT_PROMPT)

# put it all together
chain = typofix_ptuner.chain | chunker_ptuner.chain
prod_chain = ChatChain(chain)

# clean the WHOLE DataFrame
df_cleaned = dcl.clean_df(prod_chain, df)
```