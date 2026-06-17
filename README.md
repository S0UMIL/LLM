# LLM Projects with LangChain

A collection of mini projects built while learning LangChain, Ollama, and Streamlit.

## Projects

### 1. AskMyCV (PDFreader.py)
A RAG-powered app that lets you query your resume using natural language.
Built with LangChain, FAISS, HuggingFace embeddings, and Llama 3.2 running locally via Ollama.

### 2. TranslatorBot (TranslatorBot.py)
A simple language translator app using LangChain prompt templates and Llama 3.2.

### 3. FootballerHub (footballerHub.py)
A Streamlit app that fetches footballer info and DOB using LangChain prompt chaining.

### 4. AskMyCV-RAG (test.ipynb)
A RAG-based chatbot that lets you query your resume using natural language, built with LangChain, Pinecone vector database, HuggingFace embeddings, and Llama 3.2 via Ollama.

### 5.Automated design/project report generator(RAGproject.py)
Automates design/technical project report creation: a coordinator fills out an intake form, a local LLM (via Ollama) generates a structured draft, and it's auto-formatted into a Word document for review. Goal: cut report drafting from hours to minutes with consistent formatting.
Current stage: basic form → LLM → docx pipeline working. Next: company template formatting, proper heading structure, domain-specific prompt tuning.

## Tech Stack
- LangChain
- Ollama (Llama 3.2)
- Streamlit
- FAISS
- HuggingFace Embeddings
- Pinecone
- python-dotenv
- python-docx
