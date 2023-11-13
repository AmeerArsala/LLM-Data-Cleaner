# For making more readable doc strings as prompts
def cut_ends(text: str):
    return text[1:-1]


class ChatIO:
    def __init__(self, input: str, output: str):
        super()
        self.input = input
        self.output = output

    def as_dict(self, input_name: str, output_name: str):
        return {input_name: self.input, output_name: self.output}
    
    # Assume chat_ios is a list of ChatIO
    def make_chat(chat_ios: [], input_name: str, output_name: str):
        return [chat_io.as_dict(input_name, output_name) for chat_io in chat_ios]