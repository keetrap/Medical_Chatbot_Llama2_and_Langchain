# Medical_Chatbot_Llama2_and_Langchain

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/keetrap/Medical_Chatbot_Llama2_and_Langchai
```

### STEP 01- Create a virtual environment after opening the repository

```bash
python -m venv myenv
```

```bash
./myenv/Scripts/activate 
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open localhost:5000
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone