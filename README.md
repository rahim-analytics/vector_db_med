🩺 MediBot – AI-Powered Medical Chatbot

MediBot is an AI chatbot that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses to medical queries.
It uses LangChain for orchestration, Pinecone as a vector database, and Streamlit for an interactive UI.

🚀 Features

📄 Upload and extract text from PDF medical documents (via PyPDF2)

✂️ Text chunking for efficient retrieval

📊 Store embeddings in Pinecone (online vector database)

🔍 Semantic search for relevant medical knowledge

🤖 RAG-based chatbot responses using LangChain

🌐 Simple and interactive UI built with Streamlit

🏗️ Project Architecture

Text Extraction → Extracts raw text from uploaded PDFs.

Chunking & Embeddings → Breaks text into chunks and generates embeddings.

Database Storage → Stores embeddings in Pinecone for vector search.

Semantic Search → Finds the most relevant chunks based on user queries.

RAG Pipeline → Combines retrieved context with LLM to generate accurate answers.

Frontend → Streamlit app for easy interaction with MediBot.

🛠️ Technologies Used

LangChain
 – RAG pipeline & LLM orchestration

Pinecone
 – Vector database for semantic search

PyPDF2
 – PDF text extraction

Streamlit
 – Web UI framework

OpenAI API / LLMs – For generating chatbot responses
