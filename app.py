import os
import streamlit as st
import bs4
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG with Memory", layout="wide")
st.title("🔎 RAG Application with Gemini + Chroma")

# Input Google API Key
api_key = st.text_input("Enter your Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # -----------------------
    # Initialize embeddings & model
    # -----------------------
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True)

    # -----------------------
    # Load documents from a webpage
    # -----------------------
    url = st.text_input("Enter a webpage URL", 
                        value="https://lilianweng.github.io/posts/2023-06-23-agent/")

    if st.button("Load & Process Document"):
        with st.spinner("Loading and processing document..."):
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
            )
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
            retriever = vectorstore.as_retriever()

            # -----------------------
            # Build RAG chain
            # -----------------------
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say you don't know. "
                "Keep answers concise (max 3 sentences).\n\n{context}"
            )

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            qa_chain = create_stuff_documents_chain(model, chat_prompt)
            rag_chain = create_retrieval_chain(retriever, qa_chain)

            # -----------------------
            # Chat Section
            # -----------------------
            st.subheader("💬 Ask Questions")
            user_input = st.text_input("Your question:")

            if user_input:
                with st.spinner("Generating answer..."):
                    response = rag_chain.invoke({"input": user_input})
                    st.write("### Answer:")
                    st.success(response["answer"])
