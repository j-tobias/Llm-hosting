from fastapi import FastAPI


app = FastAPI()





#The Interface to send Data to the Model
@app.post("/chat/")
async def recieve(data: dict):
    """
    {
    "prompt":"Write a poem about a cat.",
    "max_length": 30,
    "context": ""
    }
    """

    print(data)

    # retrieve the Prompt from the sended Data (and strip it -> this is recommended by Meta)
    prompt = data.get("prompt")
    prompt.strip()

    # retrieve the max length of the generated Text
    max_len = data.get("max_length")

    # get the context if there is one
    context = data.get("context")

    response = Model.generate(prompt, max_len, context)
    
    # return the generated text
    return response




if __name__ == "__main__":
    import uvicorn
    from Model import LLM

    print("Step 1: Configure IP & Port")
    # Configure the API / Server
    ip = "10.220.9.82" #has to be configured
    port_n = 5500
    
    print("Step 2: Start loading Model & Tokenizer")
    # Configure the Model
    global Model
    Model = LLM()
    

    uvicorn.run(app, host=ip, port=port_n)