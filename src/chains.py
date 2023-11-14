from langchain.llms.base import LLM


# Base class; not really meant to be used alone
class ChatChainInterface:
    def __init__(self, input_name: str):
        super().__init__()
        self.input_name = input_name
        self.chain = None
    
    def run_chat(self, text: str):
        return self.chain.invoke({self.input_name: text})


# For production
class ChatChain(ChatChainInterface):
    def __init__(self, chain, input_name: str):
        super().__init__(input_name)
        self.chain = chain