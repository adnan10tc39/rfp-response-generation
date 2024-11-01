
# RFP Response Generation App

## Prerequisites
- Python 3.11.5
- Conda

## Setup Instructions

### 1. Create the Environment
To create a new environment with Python 3.11.5, run:
```bash
conda create -p rfp_env python==3.11.5 -y
```

### 2. Edit the `.env` File
Before running the code, edit the `.env` file to include your API keys.
  e.g OPENAI_API_KEY, LLAMA_CLOUD_API_KEY
  
### 3. Add Data
Place your PDF files into the `data` folder for processing.

### 4. Run the Code
- To test the setup, run:
  ```bash
  python main.py
  ```
- To start the chatbot, run:
  ```bash
  python app1.py
  ```

## Notes
- Make sure the environment is activated before running the scripts:
  ```bash
  conda activate rfp_env
  ```
- Install the required dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
