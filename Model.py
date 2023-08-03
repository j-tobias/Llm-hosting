import os
from transformers import LlamaForCausalLM, LlamaTokenizer

class LLM:

    def __init__ (self):
        # retrieve the path to the model
        model_path = os.path.join(os.getcwd(), "Llama-2-7b-chat-hf")
        # initiate the model and the tokenizer
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        # create attributes for the Instruct and Context Tokens
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_CONT, self.E_CONT = "[CONTEXT]", "[/CONTEXT]"

        # create the Default System Prompt
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
    
    def get_prompt(self, instruction):
        """
        Generates the Prompt which includes the System Prompt to guide the Models behaviour
        
        The Structure is: SYSTEM_PROMPT + INSTRUCTION
        """
        prompt_template =  self.B_INST + self.SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template
    
    def get_prompt_with_context(self, instruction, context):
        """
        Generates the Prompt which includes the System Prompt to guide the Models behaviour and also adds context

        The Structure is: SYSTEM_PROMPT + CONTEXT + INSTRUCTION
        """
        prompt_template =  self.B_INST + self.SYSTEM_PROMPT + self.B_CONT + context + self.E_CONT + instruction + self.E_INST
        return prompt_template
    
    def cut_off_text(self, text:str, prompt:str):
        """
        The model repeats the original prompt in it's response and the user should not see that
        """
        index = text.find(prompt)
        text = text[:index] if index != -1 else text
        return text

    def remove_substring(self, string: str, substring):
        return string.replace(substring, "")

    def generate(self, text, length, context = None):
        """
        Generates a response to the input text
        """
        # get the prompt
        prompt = self.get_prompt(text) if context == None else self.get_prompt_with_context(text, context)
        # get the input ids
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # hand input over to model and generate the response
        outputs = self.model.generate(**inputs, 
                                        max_new_tokens = length,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        )
        # clean up the answer
        final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = self.cut_off_text(final_outputs, '</s>')
        final_outputs = self.remove_substring(final_outputs, prompt)
        final_outputs = self.remove_substring(final_outputs, self.SYSTEM_PROMPT) 

        return final_outputs