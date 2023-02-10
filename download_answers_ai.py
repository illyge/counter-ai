import jsonlines
from util.openai_util import fetch_answer

questions_filename = "1673895202.jsonl"


def download_answers_ai():
    with jsonlines.open(f"./data/gpt_api/questions.jsonl", "r") as questions_reader:
        questions = list(questions_reader)
        print(f"Loaded {len(questions)} questions")

    try:
        with jsonlines.open(f"./data/raw/answers_gpt_api/answers_ai.jsonl", "r") as answers_reader:
            answers = list(answers_reader)
            print(f"Loaded {len(answers)} answers")
    except FileNotFoundError:
        answers = []

    unanswered = [q for q in questions if q['question_id'] not in [a['question_id'] for a in answers]]
    print(f"Unanswered questions {len(unanswered)}")
    tokens_spent = 0
    with jsonlines.open(f"./data/raw/answers_gpt_api/answers_ai.jsonl", "a") as output:
        for index, question in enumerate(unanswered):
            print(f"Processing question_id {question['question_id']} at {index}. Total tokens spent: {tokens_spent}. "
                  f"Money spent {tokens_spent*0.00002:.2f}")
            try:
                answer = fetch_answer(question['question'])
                tokens_spent += answer['tokens']
                output.write(
                    {
                        'question_id': question['question_id'],
                        'answer': answer['answer'],
                        'tokens_spent': answer['tokens']
                    })
            except Exception as e:
                print(f"Error for {question['question_id']}. Exception {e}")
                print(f"Skipping index {index}")
    print(f"Total tokens spent {tokens_spent}")


if __name__ == "__main__":
    download_answers_ai()