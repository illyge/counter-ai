import streamlit as st
import requests

def send_request(text1, text2):
    # Replace with the url of your REST API
    url = "http://counter-ai.illyge.com/classify"

    # Send a POST request to the API with the input text
    response = requests.post(url, json={"question": text1, "answer": text2})

    # Return the response text
    return response

# Create the text input areas
text1 = st.text_area("Enter Stackoverflow question")
text2 = st.text_area("Enter answer (from Stackoverflow or AI-generated")

# Send the request and display the result
if st.button("Submit"):
    spinner = st.spinner("Loading...")
    result = send_request(text1, text2)
    if result.status_code == 200:
        json_result = result.json()
        if type(json_result) is dict and 'ai_generated' in json_result.keys():
            st.success(json_result)
            st.success("It's generated by GPT" if json_result['ai_generated'][0] else "It's a human-generated answer")
    else:
        st.error(result.text)
