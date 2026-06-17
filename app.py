import streamlit as st

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
            answer = ask_bot(question)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
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