# Llm - Hosting

---

This Project completes the PersonaLlm Project where it was my Goal to create a self Hosted ChatBot. The ChatBot from that Project (PersonaLlm) is set up to work with the API provided by this Project (Llm - Hosting).

In this project, I aim to test the capabilities of the Llama2-7b-chat-hf model by integrating it with the PersonaLlm ChatBot through the Llm - Hosting API. The initial ChatBot was created in the PersonaLlm Project and now serves as a foundation to leverage the powerful language processing abilities of Llama2-7b-chat-hf. I'll evaluate the model's performance, optimize responses, and document the results to develop a more sophisticated self-hosted ChatBot. The integration marks a significant advancement in natural language processing and brings us closer to creating an intuitive AI chat companion.

## Structure

---

### main.py

The main.py file contains the API and this file will run the uvicorn server. In this file please configure the IP and the Port to suit your System

### Model.py

The [Model.py](http://Model.py) file contains the custom class which handles the model and it’s tokenizer. I have also made use of the <<SYS>>  and [INST] token to creat the SystemPrompt as adviced by Meta in the [official Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) 

## Set-up

---

This project is not intended do be replicated. But if you want to try it on your own i would recommend the following steps.

1. Download the Llama2-7b-chat-hf model from Hugginface → [link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. clone this repository
    
    ```powershell
    git -clone https://github.com/j-tobias/Llm-hosting
    ```
    
3. create a Folder in this repository called “Llama-2-7b-chat-hf”. In there copy all files downloaded from **Hugging Face** 🤗

**Requirements**

The following python libraries will be neccesary

- transformers
- fastapi
- uvicorn
