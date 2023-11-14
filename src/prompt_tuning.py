from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from prompts.prompt_formatting import ChatIO, FewShotPrompt
from chains import ChatChainInterface


def as_template(name: str):
    return "".join(["{", name, "}"])


"""
To chain with LCEL: let's say you have a PromptTuner object called prompt_tuner
the chain would be like so:

chain = prompt_tuner.chain | prompt_tuner2.chain | ...

and you get the idea
"""
class PromptTuner(ChatChainInterface):
    def __init__(self, llm: LLM, few_shot_prompt: FewShotPrompt, input_name="input", output_name="output"):
        super().__init__(input_name)
        self.llm = llm
        self.few_shot_prompt = few_shot_prompt
        self.output_name = output_name

        # Few Shot Learning
        example_few_shot_prompt = ChatPromptTemplate.from_messages([
            ("human", as_template(input_name)),
            ("ai", as_template(output_name))
        ])

        few_shot_examples_dict = ChatIO.make_chat(few_shot_prompt.examples, input_name, output_name)
        few_shot_chat_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_few_shot_prompt, examples=few_shot_examples_dict)

        # Final Chat Prompt: incorporate the few-shot learning
        self.chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", few_shot_prompt.sys_initial_prompt),
            few_shot_chat_prompt,
            ("human", as_template(input_name))
        ])

        self.chain = (self.chat_prompt_template | self.llm)

"""
Usage example to clean all columns:

import pandas as pd

df = pd.read_csv("my_dataset.csv")

import prompts.typofix as typofix
import prompts.chunker as chunker
from chains import ChatChain
from data_cleaning import clean_df

typofix_ptuner = PromptTuner(ChatAnthropic(), typofix.FEW_SHOT_PROMPT)
chunker_ptuner = PromptTuner(ChatAnthropic(), chunker.FEW_SHOT_PROMPT)

# put it all together
chain = typofix_ptuner.chain | chunker_ptuner.chain
prod_chain = ChatChain(chain)

# clean the WHOLE DataFrame
df_cleaned = clean_df(prod_chain, df)
"""

