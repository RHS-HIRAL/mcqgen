import os
import PyPDF2
import json
import traceback
from src.mcqgenerator.logger import logging


def read_file(file):
    logging.info("Started reading the uploaded file...")
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()
            logging.info("PDF file read successfully.")
            return text

        except Exception as e:
            logging.error("Error reading the PDF file.", exc_info=True)
            raise Exception("Error reading the PDF file.")

    elif file.name.endswith(".txt"):
        logging.info("Text file read successfully.")
        return file.read().decode("utf-8")

    else:
        logging.error("Unsupported file format uploaded.")
        raise Exception(
            "Unsupported file format, only pdf and text files are supported."
        )


def get_table_data(quiz):
    try:
        logging.info("Formatting quiz data for table display...")
        quiz_table_data = []
        for key, value in quiz.items():
            mcq = value["mcq"]
            options = " | ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
        logging.info("Table data formatted successfully.")
        return quiz_table_data

    except Exception as e:
        logging.error("Error formatting table data.", exc_info=True)
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
