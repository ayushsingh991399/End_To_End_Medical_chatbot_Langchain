import streamlit as st
import os
from dotenv import load_dotenv

from utility.utility import huggingface_embeddings, init_pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.sidebar.title("Settings")
st.sidebar.info("Currently using existing PDFs in Pinecone for Q&A.")

with st.spinner("Initializing embeddings and Pinecone..."):
    embeddings = huggingface_embeddings()
    vector_store = init_pinecone(INDEX_NAME, embeddings, PINECONE_API_KEY)


st.title("🤖 Medical PDF Chatbot")
st.write("Ask questions based on the existing PDFs indexed in Pinecone.")

with st.form(key="chat_form"):
    user_question = st.text_input("Type your question:")
    submit_button = st.form_submit_button("Send")


if submit_button and user_question:
    with st.spinner("Generating answer..."):
        prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Make sure your answer is at least 20 words long.

Context: {context}
Question: {question}

Helpful answer:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        result = qa({"query": user_question})


        st.session_state.chat_history.append({
            "question": user_question,
            "answer": result['result']
        })


st.subheader("Chat History")
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")

