from unittest import result

import streamlit as st


import os
import streamlit as st




from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 

@st.cache_resource
def load_vectorstore():
    loader = PyPDFDirectoryLoader("zyro-dynamics-hr-corpus")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20
    }
)
import os
from dotenv import load_dotenv

load_dotenv()

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro_rag_challenge"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


from langchain_groq import ChatGroq



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics.

Answer the question using ONLY the provided context.

If the answer is not found in the context, say:
"I could not find that information in the HR policy documents."

Context:
{context}

Question:
{question}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_chain(question: str):
    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    sources = []

    for doc in docs:
        if "source" in doc.metadata:
            sources.append(doc.metadata["source"])

    return {
        "answer": answer,
        "sources": list(set(sources))
    }



OOS_PROMPT = """
You are a classifier.

Determine if the question is related to Zyro Dynamics HR policies,
employees, leave, attendance, payroll, benefits, conduct, onboarding,
security, travel, work-from-home, or company procedures.

Respond only with YES or NO.
"""


REFUSAL_MESSAGE = (
    "Sorry, I can only answer questions related to Zyro Dynamics HR policies and procedures."
)


def ask_bot(question: str):
    decision = llm.invoke(
        f"{OOS_PROMPT}\n\nQuestion: {question}"
    ).content.strip().upper()

    if "NO" in decision:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    return rag_chain(question)


 



st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.caption("Ask questions about company HR policies, onboarding, travel, leave, remote work, and employee procedures.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an HR policy question...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):
           result = ask_bot(question)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📄 Sources"):
                for source in result["sources"]:
                    st.write(source)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )

with st.sidebar:

    st.header("📌 About")

    st.write(
    '''
    This assistant uses:

    • LangChain RAG Pipeline
    • FAISS Vector Store
    • HuggingFace Embeddings
    • Groq LLM
    • LangSmith Tracing
    '''
)

    st.header("💡 Example Questions")

    st.markdown('''
    - What is the work from home policy?
    - How does employee onboarding work?
    - What expenses can be reimbursed?
    - What is the leave approval process?
    - What are the remote work security requirements?
    ''')
