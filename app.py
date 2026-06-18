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
    search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7}
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
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1024
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics.

Important:
In some documents the company is referred to as Acrux Dynamics.
Treat Zyro Dynamics and Acrux Dynamics as the same company.

Answer ONLY using information explicitly present in the provided context.

Rules:

1. Do NOT use outside knowledge.
2. Do NOT make assumptions.
3. Do NOT infer missing information.
4. Do NOT add explanations that are not supported by the context.
5. Do NOT invent policies, dates, numbers, eligibility criteria, or procedures.
6. If information is not present in the context, respond exactly:

I could not find that information in the HR policy documents.

7. For multi-part questions:
   - Answer each part separately.
   - Only answer parts supported by the context.
   - If a part is not supported, state that it could not be found.

8. Keep answers concise and factual.

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
You are a strict binary classifier for an HR assistant.

The company may be referred to as:
- Zyro Dynamics
- Acrux Dynamics

Treat both names as the same company.

Return YES if the question is about:
- HR policies
- Leave
- Attendance
- Payroll
- Compensation
- Benefits
- Performance reviews
- Promotions
- Employee conduct
- Onboarding
- Offboarding
- Travel and expenses
- Work from home
- IT security policies
- Recruitment and hiring
- Company procedures described in HR documents

Return NO if the question:
- Is unrelated to HR policies
- Asks for gaming, coding, education, health, finance, or general knowledge advice
- Asks about another company's policies
- Compares Zyro/Acrux Dynamics with another company
- Asks about company revenue, profits, customers, products, market position, or business performance
- Requests information not covered by HR documentation

Respond with EXACTLY one word:

YES
or
NO

No explanation.
No punctuation.

Examples:

Q: What is the leave approval process?
A: YES

Q: How does employee onboarding work?
A: YES

Q: What are the remote work security requirements?
A: YES

Q: Tell me the best strategy to rank up in Valorant.
A: NO

Q: What was Acrux Dynamics revenue last year?
A: NO

Q: Compare this company's leave policy with Zoho.
A: NO

Q: How does AcruxCRM compare to Salesforce?
A: NO
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
