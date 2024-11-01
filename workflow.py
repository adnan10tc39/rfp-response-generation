from typing import List
from pydantic import BaseModel
from pathlib import Path
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.llms import LLM
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document
import logging
import json
from llama_index.llms.openai import OpenAI
import ollama
from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


# this is the research agent's system prompt, tasked with answering a specific question
AGENT_SYSTEM_PROMPT = """\
You are a research agent tasked with filling out a specific form key/question with the appropriate value, given a bank of context.
You are given a specific form key/question. Think step-by-step and use the existing set of tools to help answer the question.

You MUST always use at least one tool to answer each question. Only after you've determined that existing tools do not \
answer the question should you try to reason from first principles and prior knowledge to answer the question.

You MUST try to answer the question instead of only saying 'I dont know'.

"""

# This is the prompt tasked with extracting information from an RFP file.
EXTRACT_KEYS_PROMPT = """\
You are provided an entire RFP document, or a large subsection from it. 

We wish to generate a response to the RFP in a way that adheres to the instructions within the RFP, \
including the specific sections that an RFP response should contain, and the content that would need to go \
into each section.

Your task is to extract out a list of "questions", where each question corresponds to a specific section that is required in the RFP response.
Put another way, after we extract out the questions we will go through each question and answer each one \
with our downstream research assistant, and the combined
question:answer pairs will constitute the full RFP response.

- Make sure the questions are comprehensive and adheres to the RFP requirements.
- Make sure each question is descriptive - this gives our downstream assistant context to fill out the value for that question 
- Extract out all the questions as a list of strings.

"""

# this is the prompt that generates the final RFP response given the original template text and question-answer pairs.
GENERATE_OUTPUT_PROMPT = """\
You are an expert analyst.
Your task is to generate an RFP response according to the given RFP and question/answer pairs.

You are given the following RFP and qa pairs:

<rfp_document>
{output_template}
</rfp_document>

<question_answer_pairs>
{answers}
</question_answer_pairs>

Not every question has an appropriate answer. This is because the agent tasked with answering the question did not have the right context to answer it.
If this is the case, you MUST come up with an answer that is reasonable. You CANNOT say that you are unsure in any area of the RFP response.


Please generate the output according to the template and the answers, in markdown format.
Directly output the generated markdown content, do not add any additional text, such as "```markdown" or "Here is the output:".
Follow the original format of the template as closely as possible, and fill in the answers into the appropriate sections.
"""




_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class OutputQuestions(BaseModel):
    questions: List[str]

class OutputTemplateEvent(Event):
    docs: List[Document]

class QuestionsExtractedEvent(Event):
    questions: List[str]

class HandleQuestionEvent(Event):
    question: str

class QuestionAnsweredEvent(Event):
    question: str
    answer: str

class CollectedAnswersEvent(Event):
    combined_answers: str

class LogEvent(Event):
    msg: str
    delta: bool = False

class RFPWorkflow(Workflow):
    def __init__(
        self,
        tools,
        parser,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = "data_out_rfp",
        agent_system_prompt: str = AGENT_SYSTEM_PROMPT,
        generate_output_prompt: str = GENERATE_OUTPUT_PROMPT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tools = tools
        self.parser = parser
        # self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.output_dir = output_dir
        self.agent_system_prompt = agent_system_prompt

        out_path = Path(self.output_dir) / "workflow_output"
        out_path.mkdir(parents=True, exist_ok=True)
        self.generate_output_prompt = PromptTemplate(generate_output_prompt)

    @step
    async def parse_output_template(
        self, ctx: Context, ev: StartEvent
    ) -> OutputTemplateEvent:
        out_template_path = Path(f"{self.output_dir}/workflow_output/output_template.jsonl")
        if out_template_path.exists():
            with open(out_template_path, "r") as f:
                docs = [Document.model_validate_json(line) for line in f]
        else:
            docs = await self.parser.aload_data(ev.rfp_template_path)
            with open(out_template_path, "w") as f:
                for doc in docs:
                    f.write(doc.model_dump_json())
                    f.write("\n")

        await ctx.set("output_template", docs)
        return OutputTemplateEvent(docs=docs)

    @step
    async def extract_questions(
        self, ctx: Context, ev: OutputTemplateEvent
    ) -> HandleQuestionEvent:
        docs = ev.docs
        out_keys_path = Path(f"{self.output_dir}/workflow_output/all_keys.txt")
        if out_keys_path.exists():
            with open(out_keys_path, "r") as f:
                output_qs = [q.strip() for q in f.readlines()]
        else:
            all_text = "\n\n".join([d.get_content(metadata_mode="all") for d in docs])
            prompt = PromptTemplate(template=EXTRACT_KEYS_PROMPT)
            try:
                output_qs = self.llm.structured_predict(
                    OutputQuestions, prompt, context=all_text
                ).questions
            except Exception as e:
                _logger.error(f"Error extracting questions from page: {all_text}")
                _logger.error(e)

            with open(out_keys_path, "w") as f:
                f.write("\n".join(output_qs))

        await ctx.set("num_to_collect", len(output_qs))
        for question in output_qs:
            ctx.send_event(HandleQuestionEvent(question=question))
        return None

    @step
    async def handle_question(
        self, ctx: Context, ev: HandleQuestionEvent
    ) -> QuestionAnsweredEvent:
        question = ev.question
        research_agent = FunctionCallingAgentWorker.from_tools(
            self.tools, llm=self.llm, verbose=False, system_prompt=self.agent_system_prompt
        ).as_agent()
        response = await research_agent.aquery(question)

        if self._verbose:
            msg = f">> Asked question: {question}\n>> Got response: {str(response)}"
            ctx.write_event_to_stream(LogEvent(msg=msg))

        return QuestionAnsweredEvent(question=question, answer=str(response))

    @step
    async def combine_answers(
        self, ctx: Context, ev: QuestionAnsweredEvent
    ) -> CollectedAnswersEvent:
        num_to_collect = await ctx.get("num_to_collect")
        results = ctx.collect_events(ev, [QuestionAnsweredEvent] * num_to_collect)
        if results is None:
            return None

        combined_answers = "\n".join([result.model_dump_json() for result in results])
        with open(f"{self.output_dir}/workflow_output/combined_answers.jsonl", "w") as f:
            f.write(combined_answers)

        return CollectedAnswersEvent(combined_answers=combined_answers)
    



    # @step
    # async def generate_output(
    #     self, ctx: Context, ev: CollectedAnswersEvent
    # ) -> StopEvent:
    #     output_template = await ctx.get("output_template")
    #     output_template = "\n".join(
    #         [doc.get_content("none") for doc in output_template]
    #     )

    #     if self._verbose:
    #         ctx.write_event_to_stream(LogEvent(msg=">> GENERATING FINAL OUTPUT"))

    #     resp = await self.llm.astream(
    #         self.generate_output_prompt,
    #         output_template=output_template,
    #         answers=ev.combined_answers,
    #     )

    #     final_output = ""
    #     async for r in resp:
    #         final_output += r  # Accumulate each part of the response without displaying
    #     ctx.write_event_to_stream(LogEvent(msg=final_output))

    #     with open(f"{self.output_dir}/workflow_output/final_output.md", "w") as f:
    #         f.write(final_output)

    #     return StopEvent(result=final_output)
    @step
    async def generate_output(
        self, ctx: Context, ev: CollectedAnswersEvent
    ) -> StopEvent:
        # Load the output template from context
        output_template = await ctx.get("output_template")
        output_template = "\n".join(
            [doc.get_content("none") for doc in output_template]
        )

        # Log the start of output generation if verbose
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> GENERATING FINAL OUTPUT"))

        # Generate the response asynchronously
        resp = await self.llm.astream(
            self.generate_output_prompt,
            output_template=output_template,
            answers=ev.combined_answers,
        )

        # Accumulate the response to display as a single output
        final_output = ""
        async for r in resp:
            final_output += r  # Collect all parts without immediate display

        # Write the accumulated final output once to the event stream
        ctx.write_event_to_stream(LogEvent(msg=final_output))

        # Save the final output to a file
        with open(f"{self.output_dir}/workflow_output/final_output.md", "w") as f:
            f.write(final_output)

        # Return the final output as the result of this step
        return StopEvent(result=final_output)

