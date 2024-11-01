import streamlit as st
import asyncio
import nest_asyncio
from pathlib import Path
from data_loading import ensure_directories_exist, get_file_list, parser
from summarization import summarize_files, load_summaries
from vector_store import setup_vector_store
from tool_generation import generate_tools
from workflow import RFPWorkflow, LogEvent
from llama_index.llms.openai import OpenAI
# from llama_index.utils.workflow import draw_all_possible_flows
import ollama
from langchain_community.llms import Ollama
# from langchain_ollama.llms import OllamaLLM

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

async def run_rfp_workflow(file_path):
    data_dir = "data"
    data_out_dir = "data_out_rfp"
    persist_dir = "storage_rfp_chroma"

    ensure_directories_exist([data_dir, data_out_dir])

    # Loading and summarizing the file
    files = get_file_list(data_dir)
    file_dicts = {}

    if files:
        existing_summaries = load_summaries(data_out_dir)
        if existing_summaries:
            file_dicts = existing_summaries
        else:
            for f in files:
                full_file_path = str(Path(data_dir) / f)
                file_docs = parser.load_data(full_file_path)
                for idx, d in enumerate(file_docs):
                    d.metadata["file_path"] = f
                    d.metadata["page_num"] = idx + 1
                file_dicts[f] = {"file_path": full_file_path, "docs": file_docs}
            file_dicts = summarize_files(file_dicts, data_out_dir)

    index = setup_vector_store(file_dicts, persist_dir)
    tools = generate_tools(files, file_dicts, index)

    llm = OpenAI(model="gpt-4o-mini")

    workflow = RFPWorkflow(
        tools,
        parser=parser,
        llm=llm,
        verbose=True,
        timeout=None
    )

    # draw_all_possible_flows(RFPWorkflow, filename="rfp_workflow.html")
    handler = workflow.run(rfp_template_path=file_path)

    response_text = ""
    try:
        async for event in handler.stream_events():
            if isinstance(event, LogEvent):
                if event.delta:
                    response_text += event.msg
                    st.write(event.msg)
                else:
                    response_text += event.msg
                    st.write(event.msg)
    except Exception as e:
        st.error(f"Error during event streaming: {e}")

    response = await handler
    return str(response)

st.title("RFP Response Generation")
uploaded_file = st.file_uploader("Select an RFP PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to 'data' directory
    with open(Path("data") / uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display response generation status and results
    st.write("Generating response...")
    
    # Run the RFP workflow asynchronously and display the result
    response = asyncio.run(run_rfp_workflow(str(Path("data") / uploaded_file.name)))
    
    st.subheader("Response:")
    st.write(response)
