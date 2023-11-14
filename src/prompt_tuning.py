from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from prompts.prompt_formatting import ChatIO
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
    def __init__(self, llm: LLM, sys_initial_prompt: str, few_shot_examples: list[ChatIO], input_name="input", output_name="output"):
        super().__init__(input_name)
        self.llm = llm
        self._sys_initial_prompt = sys_initial_prompt
        self._few_shot_examples = few_shot_examples
        self.output_name = output_name

        # Few Shot Learning
        example_few_shot_prompt = ChatPromptTemplate.from_messages([
            ("human", as_template(input_name)),
            ("ai", as_template(output_name))
        ])

        few_shot_examples_dict = ChatIO.make_chat(few_shot_examples, input_name, output_name)
        few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_few_shot_prompt, examples=few_shot_examples_dict)

        # Final Chat Prompt: incorporate the few-shot learning
        self.chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", sys_initial_prompt),
            few_shot_prompt,
            ("human", as_template(input_name))
        ])

        self.chain = (self.chat_prompt_template | self.llm)

