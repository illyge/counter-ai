import requests
import time

def fetch_questions(page, pagesize):
    url = "https://api.stackexchange.com/2.2/questions"
    params = {
        "site": "stackoverflow",
        "sort": "votes",
        "pagesize": pagesize,  # Number of questions per page
        "answers": 1,
        "filter": "!*MQIL7pRpsdq5H)nUUCB(_njhjqb",
        "key": "gJE1zbvB18v8sS7Opl43lg((",
        "page": page
    }

    count = 4
    response_code = 502
    timeouts = [60, 120, 60, 10]
    while count > 0:
        response = requests.get(url, params=params)
        response_code = response.status_code
        retry_condition = response_code == 400 and response.json()['error_id'] == 502
        time.sleep(timeouts[count-1])

        if not retry_condition:
            break
        print(f"Retrying page {page} count {count}")
        count -= 1

    if not response.ok:
        raise Exception(f"Request errored with status {response_code}. {response.json()}")

    return response.json()["items"]


def fetch_answer(answer_id):
    url = f"https://api.stackexchange.com/2.2/answers/{answer_id}"
    params = {
        "site": "stackoverflow",
        "filter": "!*sVmCjZbt5MPsJxAfYAZLOjFCfva"
    }

    response = requests.get(url, params=params)
    return response.json()["items"][0]
