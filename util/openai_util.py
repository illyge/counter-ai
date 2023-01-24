import openai


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential, retry_if_exception_type,
)


@retry(wait=wait_exponential(min=6, max=300, multiplier=2, exp_base=3), stop=stop_after_attempt(6),
       retry=retry_if_exception_type(openai.error.RateLimitError))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def fetch_answer(question):
    openai.api_key = ""
    openai.error.PermissionError
    response = completion_with_backoff(
        model="text-davinci-003",
        prompt=question,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return {
        'answer': response['choices'][0]['text'],
        'tokens': response['usage']['total_tokens']
    }
