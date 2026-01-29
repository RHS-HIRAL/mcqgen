import os
from dotenv import load_dotenv
import json
import traceback
import pandas as pd
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from src.mcqgenerator.MCQGenerator import overall_chain
from src.mcqgenerator.logger import logging

load_dotenv()
model_id = os.getenv("model_id")
logging.info("Streamlit App Started. Environment variables loaded in StreamlitAPP.py.")

file_path = r"C:\Hiral\Projects\python randoms\GenAI_Practice\project1\Response.json"
try:
    with open(file_path, "r") as file:
        response_json = file.read()
    logging.info("Response JSON template loaded.")
except FileNotFoundError:
    logging.error(f"Response JSON file not found at {file_path}")
    st.error("Configuration file missing.")

st.title("MCQs Creator Application with LangChain")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or txt file")
    input_number = st.number_input("No. of MCQs", min_value=3, max_value=50)
    input_subject = st.text_input("Insert Subject", max_chars=20)
    input_tone = st.text_input(
        "Complexity Level of Questions", max_chars=20, placeholder="Simple"
    )
    button = st.form_submit_button("Create MCQs")

    if (
        button
        and uploaded_file is not None
        and input_number
        and input_subject
        and input_tone
    ):
        logging.info("User inputs received. Starting process...")
        with st.spinner("Loading..."):
            try:
                input_text = read_file(uploaded_file)
                logging.info("Input text extracted from file.")

                with get_usage_metadata_callback() as cb:
                    logging.info("Invoking LangChain Process...")
                    result = overall_chain.invoke(
                        {
                            "text": input_text,
                            "number": input_number,
                            "subject": input_subject,
                            "tone": input_tone,
                            "response_json": json.dumps(response_json),
                        }
                    )
                logging.info(
                    f"Chain execution completed. Token metadata: {cb.usage_metadata}"
                )
                print("Token metadata: ", cb.usage_metadata.get(model_id, "N/A"))

            except Exception as e:
                logging.error("Error during Chain execution", exc_info=True)
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error occurred while generating MCQs")

            else:
                if isinstance(result, dict):
                    logging.info("Processing result dictionary.")
                    quiz = result.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            st.text_area(label="Review", value=result["review"])
                            logging.info("Results displayed to user successfully.")
                        else:
                            logging.error("Failed to process table data from quiz.")
                            st.error("Error in the table data")
                    else:
                        logging.error("Quiz key missing in result dictionary.")
                else:
                    logging.info("Result is not a dictionary.")
                    st.write(result)
