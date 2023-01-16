import requests
from util.stack_overflow_util import fetch_questions
import time
import jsonlines

def download_questions(pages):
    with jsonlines.open(f"./data/raw/questions/{int(time.time())}.jsonl", "w") as output_file:
        for page in range(1, pages+1):
            print(f"Processing page {page}")
            questions = fetch_questions(page, 100)
            output_file.write_all(questions)

if __name__ == "__main__":
    download_questions(2)