import jsonlines
import time

import requests

from util.stack_overflow_util import fetch_answer, fetch_all_answers

questions_filename = "questions.jsonl"

def download_answers():
    with jsonlines.open(f"./data/raw/answers_human/{int(time.time())}.jsonl", "w") as output:
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

nq=0
def download_all_answers():
    try:
        with jsonlines.open("./data/raw/answers_human/all_answers.jsonl", "r") as answers_reader:
            preexisting_question_ids = set([a['question_id'] for a in list(answers_reader)])
            print(f"Already have downloaded answers for {len(preexisting_question_ids)} questions")
    except FileNotFoundError:
        preexisting_question_ids = []
    global nq
    with jsonlines.open("./data/raw/answers_human/all_answers.jsonl", mode="a") as output:
        with jsonlines.open(f"./data/raw/questions/{questions_filename}", "r") as questions:
            for question in [q for q in questions if q['question_id'] not in preexisting_question_ids]:
                answers = fetch_all_answers(question['question_id'])
                output.write_all(answers)
                print(f"Saved {len(answers)} answers for question {question['question_id']}")
                nq += 1

if __name__ == "__main__":
    while True:
        try:
            download_all_answers()
        except requests.exceptions.ConnectTimeout:
            print("Connection timed out, sleeping...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Unexpected {e}")
            break
        finally:
            break
    print(f"Total {nq} questions processed")