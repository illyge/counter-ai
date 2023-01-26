import jsonlines
import time
from util.stack_overflow_util import fetch_answer

questions_filename = "1673895202.jsonl"

def download_answers():
    with jsonlines.open(f"./data/raw/answers/{int(time.time())}.jsonl", "w") as output:
        with jsonlines.open(f"./data/raw/questions/{questions_filename}", "r") as questions:
            for question in questions:
                if "accepted_answer_id" in question.keys():
                    answer_id = question["accepted_answer_id"]
                    print (f"Processing answer {answer_id}")
                    try:
                        answer = fetch_answer(answer_id)
                        output.write(answer)
                    except Exception as e:
                        print(f"Error for {answer_id}. Exception {e}")
            # for page in pages:
            #     print(f"Processing page {page}")
            #     try:
            #         questions = fetch_questions(page, 100)
            #         output_file.write_all(questions)
            #     except Exception as e:
            #         print(f"Exception for page {page}: {e}")
            #


if __name__ == "__main__":
    download_answers()
