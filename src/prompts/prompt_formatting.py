# For making more readable doc strings as prompts
def cut_ends(text: str):
    return text[1:-1]

# This class allows us to decouple the raw prompts and responses to how we would be using it (i.e. no IO template names to actively use, etc.)
class ChatIO:
    def __init__(self, input_text: str, output_text: str):
        super().__init__()
        self.input_text = input_text
        self.output_text = output_text

    def as_dict(self, input_name: str, output_name: str):
        return {input_name: self.input_text, output_name: self.output_text}
    
    # Assume chat_ios is a list of ChatIO
    def make_chat(chat_ios: list[ChatIO], input_name: str, output_name: str):
        return [chat_io.as_dict(input_name, output_name) for chat_io in chat_ios]


class FewShotPrompt:
    def __init__(self, sys_initial_prompt: str, few_shot_examples: list[ChatIO]):
        super().__init__()
        self.sys_initial_prompt = sys_initial_prompt
        self.examples = few_shot_examples