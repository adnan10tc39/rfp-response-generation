# import streamlit as st
# import nest_asyncio
# import asyncio
# from pathlib import Path
# import base64
# from workflow import RFPWorkflow, LogEvent
# from data_loading import ensure_directories_exist, get_file_list, parser
# from summarization import summarize_files, load_summaries
# from vector_store import setup_vector_store
# from tool_generation import generate_tools
# from llama_index.llms.openai import OpenAI


# # Apply nest_asyncio to allow nested event loops if necessary
# nest_asyncio.apply()

# # Function to display PDF
# @st.cache_data
# def display_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # Function to run RFP response generation workflow
# async def generate_rfp_response(file_path, output_dir, data_dir="data", persist_dir="storage_rfp_chroma"):
#     st.write("Initializing directories...")
#     ensure_directories_exist([data_dir, output_dir])
#     file_name = Path(file_path).name

#     st.write("Loading the file and creating the input for the workflow...")
#     file_dicts = {}
#     full_file_path = str(Path(data_dir) / file_name)
#     file_docs = parser.load_data(full_file_path)
#     st.write(f"Loaded {len(file_docs)} documents.")

#     for idx, d in enumerate(file_docs):
#         d.metadata["file_path"] = file_name
#         d.metadata["page_num"] = idx + 1
#     file_dicts[file_name] = {"file_path": full_file_path, "docs": file_docs}
    
#     # Check for existing summaries or generate them
#     st.write("Checking for existing summaries...")
#     existing_summaries = load_summaries(output_dir)
#     if existing_summaries and file_name in existing_summaries:
#         st.write("Using existing summary for the document.")
#         file_dicts = existing_summaries
#     else:
#         st.write("Generating summary for the document...")
#         file_dicts = summarize_files(file_dicts, output_dir)
#         st.write("Summary generated and saved.")

#     # Ensure the summary is present before proceeding
#     if "summary" not in file_dicts[file_name]:
#         st.error(f"Summary for {file_name} is missing. Aborting.")
#         return "Error: Missing summary.", "Error"

#     # Set up the vector store and tools
#     st.write("Setting up the vector store and generating tools...")
#     index = setup_vector_store(file_dicts, persist_dir)
#     tools = generate_tools([file_name], file_dicts, index)

#     st.write("Initializing the workflow...")
#     llm = OpenAI(model="gpt-4o")
#     workflow = RFPWorkflow(
#         tools=tools,
#         parser=parser,
#         llm=llm,
#         verbose=True,
#         output_dir=output_dir
#     )

#     st.write("Running the workflow...")
#     handler = workflow.run(rfp_template_path=full_file_path)
#     response_text = ""
#     async for event in handler.stream_events():
#         if isinstance(event, LogEvent):
#             response_text += event.msg + "\n"
#             st.write(event.msg)  # Log each event message to Streamlit for real-time feedback

#     st.write("Retrieving the final RFP response...")
#     response = await handler
#     return response_text, str(response)

# # Streamlit app setup
# st.set_page_config(layout="wide")
# def main():
#     st.title("RFP Response Generation App")

#     uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
#     output_dir = "data_out_rfp"
#     workflow_html_path = Path(output_dir) / "workflow_output" / "rfp_workflow.html"

#     if uploaded_file is not None:
#         file_path = f"data/{uploaded_file.name}"
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
        
#         col1, col2 = st.columns(2)

#         with col1:
#             st.info("Uploaded File")
#             display_pdf(file_path)

#         if st.button("Generate RFP Response"):
#             st.info("Generating RFP Response, please wait...")

#             # Run the RFP response generation as a background task
#             loop = asyncio.get_event_loop()
#             response_task = loop.create_task(generate_rfp_response(file_path, output_dir))
#             response_text, final_response = loop.run_until_complete(response_task)
#             # response_text, final_response = asyncio.run(generate_rfp_response(file_path, output_dir))


#             with col2:
#                 st.info("RFP Response Generation Complete")
#                 st.success(final_response)
        
#         # Display the RFP workflow visualization if it exists
#         if workflow_html_path.exists():
#             st.subheader("RFP Workflow Visualization")
#             with open(workflow_html_path, "r") as f:
#                 workflow_html = f.read()
#             st.markdown(workflow_html, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

#  *********************second code is also working ********************

# import streamlit as st
# import nest_asyncio
# import asyncio
# from pathlib import Path
# import base64
# from workflow import RFPWorkflow, LogEvent
# from data_loading import ensure_directories_exist, get_file_list, parser
# from summarization import summarize_files, load_summaries
# from vector_store import setup_vector_store
# from tool_generation import generate_tools
# from llama_index.llms.openai import OpenAI

# # Apply nest_asyncio to allow nested event loops if necessary
# nest_asyncio.apply()

# # Function to display PDF
# @st.cache_data
# def display_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # Function to run RFP response generation workflow
# async def generate_rfp_response(file_path, output_dir, data_dir="data", persist_dir="storage_rfp_chroma"):
#     st.write("Initializing directories...")
#     ensure_directories_exist([data_dir, output_dir])
#     file_name = Path(file_path).name

#     st.write("Loading the file and creating the input for the workflow...")
#     file_dicts = {}
#     full_file_path = str(Path(data_dir) / file_name)
#     file_docs = parser.load_data(full_file_path)
#     st.write(f"Loaded {len(file_docs)} documents.")

#     for idx, d in enumerate(file_docs):
#         d.metadata["file_path"] = file_name
#         d.metadata["page_num"] = idx + 1
#     file_dicts[file_name] = {"file_path": full_file_path, "docs": file_docs}
    
#     # Check for existing summaries or generate them
#     st.write("Checking for existing summaries...")
#     existing_summaries = load_summaries(output_dir)
#     if existing_summaries and file_name in existing_summaries:
#         st.write("Using existing summary for the document.")
#         file_dicts = existing_summaries
#     else:
#         st.write("Generating summary for the document...")
#         file_dicts = summarize_files(file_dicts, output_dir)
#         st.write("Summary generated and saved.")

#     # Ensure the summary is present before proceeding
#     if "summary" not in file_dicts[file_name]:
#         st.error(f"Summary for {file_name} is missing. Aborting.")
#         return "Error: Missing summary.", "Error"

#     # Set up the vector store and tools
#     st.write("Setting up the vector store and generating tools...")
#     index = setup_vector_store(file_dicts, persist_dir)
#     tools = generate_tools([file_name], file_dicts, index)

#     st.write("Initializing the workflow...")
#     llm = OpenAI(model="gpt-4o")
#     workflow = RFPWorkflow(
#         tools=tools,
#         parser=parser,
#         llm=llm,
#         verbose=True,
#         output_dir=output_dir
#     )

#     st.write("Running the workflow...")
#     handler = workflow.run(rfp_template_path=full_file_path)
#     response_text = ""
#     async for event in handler.stream_events():
#         if isinstance(event, LogEvent):
#             response_text += event.msg + "\n"
#             st.write(event.msg)  # Log each event message to Streamlit for real-time feedback

#     st.write("Retrieving the final RFP response...")
#     response = await handler
#     return response_text, str(response)

# # Function to display the RFP workflow HTML file
# def display_rfp_workflow_html(workflow_html_path):
#     with open(workflow_html_path, "r") as f:
#         workflow_html = f.read()
#     st.markdown(workflow_html, unsafe_allow_html=True)

# # Streamlit app setup
# st.set_page_config(layout="wide")

# def main():
#     st.title("RFP Response Generation App")

#     uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
#     output_dir = "data_out_rfp"

#     # Define the correct path to the HTML file in the current directory
#     workflow_html_path = Path(__file__).parent / "rfp_workflow.html"

#     if uploaded_file is not None:
#         file_path = f"data/{uploaded_file.name}"
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
        
#         col1, col2 = st.columns(2)

#         with col1:
#             st.info("Uploaded File")
#             display_pdf(file_path)

#         if st.button("Generate RFP Response"):
#             st.info("Generating RFP Response, please wait...")

#             # Run the RFP response generation as a background task
#             loop = asyncio.get_event_loop()
#             response_task = loop.create_task(generate_rfp_response(file_path, output_dir))
#             response_text, final_response = loop.run_until_complete(response_task)

#             with col2:
#                 st.info("RFP Response Generation Complete")
#                 st.success(final_response)
        
#         # Display the RFP workflow visualization if the file exists
#         if workflow_html_path.exists():
#             st.subheader("RFP Workflow Visualization")
#             display_rfp_workflow_html(workflow_html_path)
#         else:
#             st.error(f"File not found: {workflow_html_path}")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import nest_asyncio
# import asyncio
# from pathlib import Path
# import base64
# import webbrowser
# from workflow import RFPWorkflow, LogEvent
# from data_loading import ensure_directories_exist, get_file_list, parser
# from summarization import summarize_files, load_summaries
# from vector_store import setup_vector_store
# from tool_generation import generate_tools
# from llama_index.llms.openai import OpenAI

# # Apply nest_asyncio to allow nested event loops if necessary
# nest_asyncio.apply()

# # Function to display PDF
# @st.cache_data
# def display_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # Function to run RFP response generation workflow
# async def generate_rfp_response(file_path, output_dir, data_dir="data", persist_dir="storage_rfp_chroma"):
#     st.write("Initializing directories...")
#     ensure_directories_exist([data_dir, output_dir])
#     file_name = Path(file_path).name

#     st.write("Loading the file and creating the input for the workflow...")
#     file_dicts = {}
#     full_file_path = str(Path(data_dir) / file_name)
#     file_docs = parser.load_data(full_file_path)
#     st.write(f"Loaded {len(file_docs)} documents.")

#     for idx, d in enumerate(file_docs):
#         d.metadata["file_path"] = file_name
#         d.metadata["page_num"] = idx + 1
#     file_dicts[file_name] = {"file_path": full_file_path, "docs": file_docs}

#     # Check for existing summaries or generate them
#     st.write("Checking for existing summaries...")
#     existing_summaries = load_summaries(output_dir)
#     if existing_summaries and file_name in existing_summaries:
#         st.write("Using existing summary for the document.")
#         file_dicts = existing_summaries
#     else:
#         st.write("Generating summary for the document...")
#         file_dicts = summarize_files(file_dicts, output_dir)
#         st.write("Summary generated and saved.")

#     # Ensure the summary is present before proceeding
#     if "summary" not in file_dicts[file_name]:
#         st.error(f"Summary for {file_name} is missing. Aborting.")
#         return "Error: Missing summary.", "Error"

#     # Set up the vector store and tools
#     st.write("Setting up the vector store and generating tools...")
#     index = setup_vector_store(file_dicts, persist_dir)
#     tools = generate_tools([file_name], file_dicts, index)

#     st.write("Initializing the workflow...")
#     llm = OpenAI(model="gpt-4o")
#     workflow = RFPWorkflow(
#         tools=tools,
#         parser=parser,
#         llm=llm,
#         verbose=True,
#         output_dir=output_dir
#     )

#     st.write("Running the workflow...")
#     response_text = ""
#     handler = workflow.run(rfp_template_path=full_file_path)
    
#     # Using asyncio to stream events in a non-blocking manner.
#     async for event in handler.stream_events():
#         if isinstance(event, LogEvent):
#             response_text += event.msg + "\n"
#             st.write(event.msg)  # Log each event message to Streamlit for real-time feedback
    
#     st.write("Retrieving the final RFP response...")
#     response = await handler
#     return response_text, str(response)

# # Function to display the RFP workflow HTML file inside the Streamlit app
# def display_rfp_workflow_html(workflow_html_path):
#     with open(workflow_html_path, "r") as f:
#         workflow_html = f.read()
#     st.markdown(workflow_html, unsafe_allow_html=True)

# # Function to open the HTML file in the browser
# def open_in_browser(html_file_path):
#     webbrowser.open_new_tab(f"file://{html_file_path}")

# # Streamlit app setup
# st.set_page_config(layout="wide")

# def main():
#     st.title("RFP Response Generation App")

#     uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
#     output_dir = "data_out_rfp"

#     # Define the correct path to the HTML file in the current directory
#     workflow_html_path = Path(__file__).parent / "rfp_workflow.html"

#     if uploaded_file is not None:
#         file_path = f"data/{uploaded_file.name}"
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
        
#         # Parallel layout for the uploaded file and chatbot response
#         col1, col2 = st.columns([1, 1])  # Use equal width for columns

#         with col1:
#             st.info("Uploaded File")
#             display_pdf(file_path)

#         with col2:
#             st.info("RFP Response")

#             if st.button("Generate RFP Response"):
#                 st.info("Generating RFP Response, please wait...")

#                 # Run the RFP response generation as a background task
#                 loop = asyncio.new_event_loop()  # Create a new event loop
#                 asyncio.set_event_loop(loop)  # Set the new loop
#                 response_task = loop.create_task(generate_rfp_response(file_path, output_dir))
#                 response_text, final_response = loop.run_until_complete(response_task)

#                 st.write(response_text)
#                 st.success(final_response)

#         # Display the RFP workflow visualization if the file exists
#         st.subheader("RFP Workflow Visualization")

#         # Add a button to open the HTML graph in a browser
#         if workflow_html_path.exists():
#             st.write("You can view the RFP Workflow graph.")
#             if st.button("View Workflow Graph in Browser"):
#                 open_in_browser(workflow_html_path)
#         else:
#             st.error(f"Workflow HTML not found: {workflow_html_path}")

# if __name__ == "__main__":
#     main()

import streamlit as st
import nest_asyncio
import asyncio
from pathlib import Path
import base64
from workflow import RFPWorkflow, LogEvent
from data_loading import ensure_directories_exist, get_file_list, parser
from summarization import summarize_files, load_summaries
from vector_store import setup_vector_store
from tool_generation import generate_tools
from llama_index.llms.openai import OpenAI
from PIL import Image
import html2image
from llama_index.utils.workflow import draw_all_possible_flows

# Apply nest_asyncio to allow nested event loops if necessary
nest_asyncio.apply()

# Function to display PDF
@st.cache_data
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def convert_html_to_image(html_file_path):
    hti = html2image.Html2Image()
    # Set the output path to the directory where the image will be saved.
    output_dir = str(html_file_path.parent)
    output_image_filename = html_file_path.stem + '.png'
    hti.output_path = output_dir
    
    # Adjust zoom level for better image clarity (higher zoom for sharper images).
    hti.browser_args = ['--force-device-scale-factor=2']
    
    # Specify the size of the image, increasing the height by 20%.
    hti.size = (hti.size[0], int(hti.size[1] * 1.2))
    
    # Capture the HTML as an image.
    hti.screenshot(html_file=str(html_file_path), save_as=output_image_filename)
    
    # Return the full path of the generated image.
    return str(Path(output_dir) / output_image_filename)




def display_rfp_workflow_image(workflow_html_path):
    image_path = convert_html_to_image(workflow_html_path)
    image = Image.open(image_path)
    st.image(image, caption="RFP Workflow Visualization", use_column_width=True)

# Function to run RFP response generation workflow
async def generate_rfp_response(file_path, output_dir, data_dir="data", persist_dir="storage_rfp_chroma"):
    st.write("Initializing directories...")
    ensure_directories_exist([data_dir, output_dir])
    file_name = Path(file_path).name

    st.write("Loading the file and creating the input for the workflow...")
    file_dicts = {}
    full_file_path = str(Path(data_dir) / file_name)
    file_docs = parser.load_data(full_file_path)
    st.write(f"Loaded {len(file_docs)} documents.")

    for idx, d in enumerate(file_docs):
        d.metadata["file_path"] = file_name
        d.metadata["page_num"] = idx + 1
    file_dicts[file_name] = {"file_path": full_file_path, "docs": file_docs}

    st.write("Checking for existing summaries...")
    existing_summaries = load_summaries(output_dir)
    if existing_summaries and file_name in existing_summaries:
        st.write("Using existing summary for the document.")
        file_dicts = existing_summaries
    else:
        st.write("Generating summary for the document...")
        file_dicts = summarize_files(file_dicts, output_dir)
        st.write("Summary generated and saved.")

    if "summary" not in file_dicts[file_name]:
        st.error(f"Summary for {file_name} is missing. Aborting.")
        return "Error: Missing summary.", "Error"

    st.write("Setting up the vector store and generating tools...")
    index = setup_vector_store(file_dicts, persist_dir)
    tools = generate_tools([file_name], file_dicts, index)

    st.write("Initializing the workflow...")
    llm = OpenAI(model="gpt-4o-mini")
    workflow = RFPWorkflow(
        tools=tools,
        parser=parser,
        llm=llm,
        verbose=True,
        output_dir=output_dir
    )

    
    draw_all_possible_flows(RFPWorkflow, filename="rfp_workflow.html")
    st.write("Running the workflow...")
    response_text = ""
    handler = workflow.run(rfp_template_path=full_file_path)
    
    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            response_text += event.msg + "\n"
            st.write(event.msg)

    st.write("Retrieving the final RFP response...")
    response = await handler
    return response_text, str(response)

# Streamlit app setup
st.set_page_config(layout="wide")

def main():
    st.title("RFP Response Generation App")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    output_dir = "data_out_rfp"

    # Define the path to the HTML file for workflow visualization
    workflow_html_path = Path(__file__).parent / "rfp_workflow.html"

    if uploaded_file is not None:
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("Uploaded File")
            display_pdf(file_path)

            # Display the workflow visualization directly below the uploaded file
            st.subheader("RFP Workflow Visualization")
            if workflow_html_path.exists():
                display_rfp_workflow_image(workflow_html_path)
            else:
                st.error(f"Workflow HTML not found: {workflow_html_path}")

        with col2:
            st.info("RFP Response")

            if st.button("Generate RFP Response"):
                st.info("Generating RFP Response, please wait...")

                # Run the RFP response generation using asyncio.run
                response_text, final_response = asyncio.run(
                    generate_rfp_response(file_path, output_dir)
                )

                st.write(response_text)
                st.success(final_response)

if __name__ == "__main__":
    main()







