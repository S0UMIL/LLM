from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
#loading pdf
pdf=PyPDFDirectoryLoader('./')
pdf_docs=pdf.load()
#splittingg into chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n",".",","," "]
)
chunks=splitter.split_documents(pdf_docs)
#getting embeddings
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store=FAISS.from_documents(chunks,embeddings)

#model
llm=OllamaLLM(model="llama3.2",temperature=0.8)
prompt=PromptTemplate(
    input_variables=['context','question'],
    template="Use the following context from {context} and answer this {question},the given context is an CV of an student try to give consise answer around 50 words and keep an good impression of the person's cv. give an overall rating of the cv out of 10 at the end of answer "
)
chain= prompt | llm
st.title("AskMyCV")
input_text=st.text_input("enter your doubts regarding cv")

questions=["What specfic skills does this candidate have?","Does the candidate havea n PHD","how many years of experience does this candidate have?"]
result=[]
ground_truths = [
    "the candidate's skills mainly covers over generative AI and Prompt engineering mainly uses tools like LangChain, FAISS, PyTorch, etc.",
    "No the candidate is currently doing undergrad",
    "currently the cv shows no work experience but has worked on multiple projects in the domain of AI"
]
for question,gt in zip(questions,ground_truths):
    docs=vector_store.similarity_search(question,k=3)
    context="\n".join([doc.page_content for doc in docs])
    answer=chain.invoke({"context":context,"question":question})
    contexts=[doc.page_content for doc in docs]
    row={"question":question,"context":context,"answer":answer,"contexts":contexts,"ground_truth":gt}
    result.append(row)
