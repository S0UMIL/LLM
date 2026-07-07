from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ---------- pipeline setup (same as PDFreader.py, minus Streamlit) ----------
pdf = PyPDFDirectoryLoader('C:/Users/soumi/OneDrive/Desktop/CODES/')
pdf_docs = pdf.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n", ".", ",", " "]
)
chunks = splitter.split_documents(pdf_docs)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embeddings)

llm = OllamaLLM(model="llama3.2", temperature=0.0)  # 0.0 here for reproducible eval runs
prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template="Use the following context from {context} and answer this {question},the given context is an CV of an student try to give consise answer around 50 words and keep an good impression of the person's cv. give an overall rating of the cv out of 10 at the end of answer "
)
chain = prompt | llm
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ---------- golden dataset ----------
questions = [
    "What specfic skills does this candidate have?",
    "Does the candidate havea n PHD",
    "how many years of experience does this candidate have?"
]
ground_truths = [
    "the candidate's skills mainly covers over generative AI and Prompt engineering mainly uses tools like LangChain, FAISS, PyTorch, etc.",
    "No the candidate is currently doing undergrad",
    "currently the cv shows no work experience but has worked on multiple projects in the domain of AI"
]

result = []
for question, ground_truth in zip(questions, ground_truths):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    contexts = [doc.page_content for doc in docs]
    answer = chain.invoke({"context": context, "question": question})
    row = {"question": question, "context": context, "answer": answer, "contexts": contexts, "ground_truth": ground_truth}
    result.append(row)
    print(f"Q: {question}\nA: {answer}\n")

# ---------- convert + evaluate ----------
eval_dataset = Dataset.from_list(result)

results = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print(results.to_pandas())
##NEW ERROR OCCURED llama 3.2 3B to SMALL to handel a task like this
