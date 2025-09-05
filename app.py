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

PINECONE_API_KEY = "Pinecone API key"
INDEX_NAME = "my-index"

# ========================
# Initialize Pinecone + RAG
# ========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="Google API Key",
    temperature=0.5
)

# Prompt template
prompt_template = """You are a helpful medical assistant. 
Use the following context to answer the question.
If you do not know, say you do not know. 
Keep answers clear and concise.

Context: {context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Global QA system
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# ========================
# Streamlit UI
# ========================
st.title("💬 MediBot Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Type your query here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Call Pinecone RAG with .invoke()
    result = qa.invoke({"query": user_query})
    answer = result["result"]
    sources = result.get("source_documents", [])

    # Show answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    # Show sources if available
    if sources:
        with st.expander("📚 Sources"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {doc.page_content[:500]}...")