from langchain.llms.base import LLM
#from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from prompts.prompt_formatting import ChatIO, FewShotPrompt
from langchain.schema.runnable.config import RunnableConfig


def ptmplt(name: str):
    return "".join(["{", name, "}"])


# Base class; not really meant to be used alone
# TODO: invoke() (agnostic) vs. predict() (raw) vs. predict_messages() (chat)
class ChatChainInterface:
    def __init__(self, input_name: str):
        super().__init__()
        self.input_name = input_name
        self.chain = None
    
    def run_chat(self, text: str):
        return self.chain.invoke({self.input_name: text})


# For production
class ChatChain(ChatChainInterface):
    def __init__(self, chain, input_name="input"):
        super().__init__(input_name)
        self.chain = chain


"""
To chain with LCEL: let's say you have a PromptTuner object called prompt_tuner
the chain would be like so:

chain = prompt_tuner.chain | prompt_tuner2.chain | ...

and you get the idea
"""
class FewShotLLM(ChatChainInterface):
    def __init__(self, llm: LLM, few_shot_prompt: FewShotPrompt, input_name="input", output_name="output", prompt_prefix="Before:\n", prompt_suffix="\n\nAfter:\n", 
                 wrap_input=""):  # example of wrap_input: `...wrap_input="{}"`
        super().__init__(input_name)
        self.llm = llm
        self.few_shot_prompt = few_shot_prompt
        self.output_name = output_name
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.wrap_input = wrap_input

        # add in delims if declared
        if len(wrap_input) > 0:
            (input_wrap_start, input_wrap_end) = (wrap_input[0], wrap_input[1])
            self.prompt_prefix = "".join([self.prompt_prefix, f"\n{input_wrap_start}"])
            self.prompt_suffix = "".join([f"\n{input_wrap_end}", self.prompt_suffix])

        # Few Shot Learning
        example_few_shot_prompt = ChatPromptTemplate.from_messages([
            ("user", self.promptify(ptmplt(input_name))),  # prefix + "{input}" + suffix
            ("assistant", ptmplt(output_name))             # "{output}"
        ])

        few_shot_examples_dict = ChatIO.make_chat(few_shot_prompt.examples, input_name, output_name)
        few_shot_chat_prompt = FewShotChatMessagePromptTemplate(  # Used to be FewShotChatMessagePromptTemplate
            example_prompt=example_few_shot_prompt,
            examples=few_shot_examples_dict#,

            # Prompt customizations
            # ? Could honestly put this above in the `example_few_shot_prompt` and below in `run_chat()` but I ain't doin that!!!
            #prefix="Before:\n",
            #suffix="\n\nAfter:"
        )

        # Final Chat Prompt: incorporate the few-shot learning
        self.chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", few_shot_prompt.sys_initial_prompt),
            few_shot_chat_prompt,
            ("user", self.promptify(ptmplt(input_name)))  # prefix + "{input}" + suffix
        ])

        self.chain = (self.chat_prompt_template | self.llm)

    def promptify(self, prompt: str):
        return "".join([self.prompt_prefix, prompt, self.prompt_suffix])

    def run_chat(self, text: str, chain_run_name="few_shot_llm"):
        #kwargs = {"stop": ["]"]}

        """chain_config = RunnableConfig(
            run_name=chain_run_name,
            tags=[],
            metadata={"llm__stop": ["]"]},
            callbacks=None,
            recursion_limit=25  # default value
        )"""

        return self.chain.invoke({self.input_name: text}) #config=chain_config)