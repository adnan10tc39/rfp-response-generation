import pickle
from llama_index.core import SummaryIndex
from llama_index.llms.openai import OpenAI
import os
import ollama
from langchain_community.llms import Ollama
# from langchain_ollama.llms import OllamaLLM

summary_llm = OpenAI(model="gpt-4o-mini")
# summary_llm = OllamaLLM(model="nemotron")

def summarize_files(file_dicts, data_out_dir):
    for f in file_dicts:
        index = SummaryIndex(file_dicts[f]["docs"])
        response = index.as_query_engine(llm=summary_llm).query(
            "Generate a short 1-2 line summary of this file to help inform an agent on what this file is about."
        )
        file_dicts[f]["summary"] = str(response)
    pickle.dump(file_dicts, open(f"{data_out_dir}/tmp_file_dicts.pkl", "wb"))
    return file_dicts

def load_summaries(data_out_dir):
    file_path = f"{data_out_dir}/tmp_file_dicts.pkl"
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))
    return None
