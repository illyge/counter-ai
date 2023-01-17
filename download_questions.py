import requests
from util.stack_overflow_util import fetch_questions
import time
import jsonlines

def download_questions(pages):
    with jsonlines.open(f"./data/raw/questions/{int(time.time())}.jsonl", "w") as output_file:
        for page in pages:
            print(f"Processing page {page}")
            try:
                questions = fetch_questions(page, 100)
                output_file.write_all(questions)
            except Exception as e:
                print(f"Exception for page {page}: {e}")

if __name__ == "__main__":
    download_questions([13, 30, 38, 43, 54, 57, 61, 66, 100])