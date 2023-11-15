import os
import dotenv
import requests

# Manual; works well with private apis
class HUGGINGFACE_APIs:
    # Authorization
    _headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}"}

    # Models
    Falcon_7b_Instruct = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    Pythia_1_4_b = "https://api-inference.huggingface.co/models/EleutherAI/pythia-1.4b" 

    def query(payload, API_URL: str):
        response = requests.post(API_URL, headers=HUGGINGFACE_APIs._headers, json=payload)
        return response.json()

    def query_llm(text: str, API_URL: str):
        return HUGGINGFACE_APIs.query({
            "inputs": text
        }, API_URL)