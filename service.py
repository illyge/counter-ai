import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel
from util.preprocess_util import prepare_data
class CounterAIApp(BaseModel):
    question: str
    answer: str

model_ref = bentoml.sklearn.get("counter-ai-model:latest")

model_runner = model_ref.to_runner()

svc = bentoml.Service("counter_ai_classifier", runners=[model_runner])


api_doc_string = r"""
Classify an answer to a StackOverflow question as an output from GPT AI or not

##### Examples

**Request**
{
   "question": "\n\n\nI'm creating a word filter that if index 1 = dog and index 2 = cat, it will return true. What should I put in next index for word?\n\nlet textContainer = ['bird', 'dog', 'cat', 'snake', 'rabbit', 'ox', 'sheep', 'tiger'];\n\nfor (let word of textContainer) {\n if (word === 'dog' && (next index for word) === 'cat') {\n return true;\n }\n}",
   
   "answer": "You can use Array.find (or Array.some)\n\nfind returns 'dog' which is not undefined (so truthy), some will return true if dog,cat is found\n\nconst textContainer = ['bird', 'dog', 'cat', 'snake', 'rabbit', 'ox', 'sheep', 'tiger'];\n\nconst found = textContainer\n .find((word, i, arr) => word==='dog' && arr[i+1] === 'cat')\nconsole.log(found); \n// if (found) return;"
}

**Response**
{
  "ai_generated": [
    false
  ]
}

**Request**
{
   "question": "How do I rename a local Git branch?\nHow do I rename a local branch which has not yet been pushed to a remote repository?\nRelated:\nRename master branch for both local and remote Git repositories\nHow do I rename both a Git local and remote branch name?\n",
   
   "answer": "To rename a local Git branch, you can use the command git branch -m old_branch_name new_branch_name. This will rename the local branch from old_branch_name to new_branch_name"
}

**Response**
{
  "ai_generated": [
    true
  ]
}

"""

@svc.api(input=JSON(pydantic_model=CounterAIApp), output=JSON(), doc=api_doc_string)
def classify(twitter_object):
    data = prepare_data(pd.DataFrame(twitter_object.dict(), index=[0]), train=False)
    prediction = model_runner.predict_proba.run(pd.DataFrame(data=data, index=[0]))
    return {
        'ai_generated': prediction[0][1] > 0.5,
        'ai_probability': prediction[0][1]
    }