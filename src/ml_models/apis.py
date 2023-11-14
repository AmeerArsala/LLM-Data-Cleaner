import os
import dotenv
import requests


class HUGGINGFACE_APIs:
    # Authorization
    _headers = {"Authorization": os.environ.get("HUGGINGFACE_API_TOKEN")}

    # Models
    Falcon_7b_Instruct = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    Pythia_1_4_b = "https://api-inference.huggingface.co/models/EleutherAI/pythia-1.4b" 

    def query(payload, API_URL: str):
        response = requests.post(API_URL, headers=_headers, json=payload)
        return response.json()

    def query_llm(text: str, API_URL: str):
        return query({
            "inputs": text
        }, API_URL)