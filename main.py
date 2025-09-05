# app.py
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from pinecone import Pinecone
from langchain.prompts import PromptTemplate

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="MediBot", page_icon="🤖")

PINECONE_API_KEY = "pcsk_DuZ3T_CjfL6WhZ6XCG5BQUJgY8yjrVpFC9MiXawjhb1L7m7zcNGQxX82QQfxWUceLT3Kt"
INDEX_NAME = "my-index"  # Use your existing Pinecone index

# ========================
# INITIALIZE (only once)
# ========================
if "qa" not in st.session_state:
    # Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key="AIzaSyAG8D6LnJlmMjJyvaKUhr5VYr27sTjNo94",
        temperature=0.5
    )

    # Prompt
    default_prompt = """You are a helpful medical assistant. 
Use the following context to answer the question.
If you don’t know, say you don’t know. 
Keep answers clear and concise.

Context: {context}
Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=default_prompt,
    )

    # Save QA chain
    rag_system = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

# ========================
# STREAMLIT APP
# ========================
st.title("💬 MediBot Assistant")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input box
user_query = st.text_input("Ask me anything:")

if user_query:
    rag_system = st.session_state["qa"]
    result = rag_system.run(user_query)

    st.session_state["chat_history"].append(
        {"user": user_query, "bot": result}
    )

# Display chat history
st.markdown("### Chat History")
for chat in st.session_state["chat_history"]:
    st.markdown(f"**🧑 You:** {chat['user']}")
    st.markdown(f"**🤖 Bot:** {chat['bot']}")
    st.markdown("---")
# Save the QA chain in session state
