import os
from os.path import join, dirname
from dotenv import load_dotenv

#dotenv_path = join(dirname(__file__), ".env")
#load_dotenv(dotenv_path)
load_dotenv()


def apply():
    load_dotenv()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")