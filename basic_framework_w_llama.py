#importing all the stuff.
import os
from langchain_ollama import OllamaLLM
import streamlit as st

##Streamlit framework

st.title('Langchain Practice using OpenAI')
input_text=st.text_input("search the topic you want")

##Llama LLMS
llm=OllamaLLM(model="llama3")

if input_text:
    st.write(llm.invoke(input_text))
  #basic framework of importing llama (when locally installed on ur device and working on top of it)
