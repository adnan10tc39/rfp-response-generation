import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

parser = LlamaParse(
    api_key=api_key,
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
)

def ensure_directories_exist(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_file_list(data_dir):
    if not os.path.exists(data_dir):
        return []
    return [f.name for f in Path(data_dir).glob('*.pdf')]
