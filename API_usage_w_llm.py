#importing all the stuff.
import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

##Streamlit framework

st.title('Langchain Practice using OpenAI')
input_text=st.text_input("search the topic you want")

##OPENAI LLMS
llm=OpenAI(temperature=0.8)

if input_text:
    st.write(llm.invoke(input_text))
  ## most basic structure for linking any ai with ur end to end application.
