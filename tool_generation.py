from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.tools import FunctionTool
from pathlib import Path

def generate_tool(file, file_description, index):
    filters = MetadataFilters(
        filters=[MetadataFilter(key="file_path", operator=FilterOperator.EQ, value=file)]
    )

    def chunk_retriever_fn(query):
        retriever = index.as_retriever(similarity_top_k=5, filters=filters)
        nodes = retriever.retrieve(query)
        return "\n\n========================\n\n".join(
            [n.get_content(metadata_mode="all") for n in nodes]
        )

    fn_name = Path(file).stem + "_retrieve"
    tool_description = f"Retrieves a small set of relevant document chunks from {file}."
    if file_description:
        tool_description += f"\n\nFile Description: {file_description}"

    return FunctionTool.from_defaults(
        fn=chunk_retriever_fn, name=fn_name, description=tool_description
    )

def generate_tools(files, file_dicts, index):
    return [generate_tool(f, file_dicts[f]["summary"], index) for f in files]
