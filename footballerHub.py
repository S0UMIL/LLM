# A simple Streamlit app using LangChain and Llama3.2 (local) to fetch 
# footballer info and DOB based on user input via prompt chaining.
#importing all the stuff.
import os
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import PromptTemplate

##Streamlit framework

st.title('Footballers HUB')
input_text=st.text_input("search the topic you want")

##prompt template
first_input=PromptTemplate(
    input_variables=['name'],
    template="tell me about footballer {name}"
)

second_input=PromptTemplate(
    input_variables=['name'],
    template="when was this {name} born"
)

##Llama LLMS
llm=OllamaLLM(model="llama3.2",temperature=0.7)
chain1= first_input | llm
chain2= second_input | llm

if input_text:
    footballer_info=(chain1.invoke({"name": input_text}))
    footballer_dob=(chain2.invoke({"name": input_text}))

    st.write(footballer_info)
    st.write(footballer_dob)


