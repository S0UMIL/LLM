import os
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import PromptTemplate

st.title("Translator GPT")
input_text=st.text_input("enter the sentence in ur known language")
target_text=st.text_input("enter the language of translation")


first_prompt=PromptTemplate(
    input_variables=['known_language','unknown_language'],
    template="In an easy way translate this {known_language} into {unknown_language}"
)

llm=OllamaLLM(model="llama3.2",temperature=0.7)
chain1=first_prompt | llm

if input_text and target_text:
    translated_language=(chain1.invoke({"known_language": input_text,"unknown_language":target_text}))
    st.write(translated_language)
