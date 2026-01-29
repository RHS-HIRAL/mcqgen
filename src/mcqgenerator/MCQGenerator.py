import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import get_usage_metadata_callback

# Load environment variables from the .env file
load_dotenv()

# Access the env variables
key = os.getenv("HF_API_KEY")

# Defining the models we might use
model_id = "Qwen/Qwen2.5-72B-Instruct"
alt_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Client
llm_client = HuggingFaceEndpoint(
    repo_id=model_id, max_new_tokens=2048, temperature=0.3, huggingfacehub_api_token=key
)

chat_model = ChatHuggingFace(llm=llm_client)

# Template
template_generation = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guid. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

# PromptTemplate for MCQ generation
gen_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template_generation,
)

# Chain 1: Generator
# This chain takes the raw inputs and outputs the Quiz JSON string
quiz_gen_chain = gen_prompt | chat_model | JsonOutputParser()

# Template for reviewing the code
template_review = """
You are an expert english grammarian and writer. Given a MCQ for {subject} students. \
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only \
use at max 50 words for complexity if the quiz is not at par with the cognitive and analytical abilities of the students, \
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student's ability.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

# PtomptTemplate for review
review_prompt = PromptTemplate(
    input_variables=["subject", "quiz"], template=template_review
)

# Chain 2: Reviewer
# This chain takes the quiz from Chain 1 and the subject, and outputs the review
review_chain = review_prompt | chat_model | StrOutputParser()

# Sequential Chain containing both the chains above
overall_chain = (
    RunnablePassthrough.assign(
        quiz=quiz_gen_chain
    )  # Runs gen_chain, stores result in 'quiz' key
    | RunnablePassthrough.assign(
        review=review_chain
    )  # Runs review_chain using 'quiz' from prev step
)
