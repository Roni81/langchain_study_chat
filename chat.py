import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title = '무엇이든 질문하세요')
st.title('무엇이든 질문하세요')



def generate_response(input_text):
    llm = ChatOpenAI(model_name = 'gpt-4o-mini', temperature = 0)
    messages = [HumanMessage(content = input_text)]
    response = llm.invoke(messages)

    st.info(response.content)

with st.form("Question"):
    text = st.text_area('질문입력:', 
                        'What types of text models does OpenAI provide?')

    submitted = st.form_submit_button("보내기")
    if submitted:
        generate_response(text)
