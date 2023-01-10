import requests

def fetch_questions_page(page, pagesize):
    url = "https://api.stackexchange.com/2.2/questions"
    params = {
        "site": "stackoverflow",
        "sort": "votes",
        "pagesize": pagesize,  # Number of questions per page
        "answers": 1,
        "filter": "!*MQIL7pRpsdq5H)nUUCB(_njhjqb",
        "page": page
    }
    response = requests.get(url, params=params)

    return response.json()["items"]

def fetch_answer(answer_id):
    url = f"https://api.stackexchange.com/2.2/answers/{answer_id}"
    params = {
        "site": "stackoverflow",
        "filter": "!*sVmCjZbt5MPsJxAfYAZLOjFCfva"
    }

    response = requests.get(url, params=params)
    return response.json()["items"][0]