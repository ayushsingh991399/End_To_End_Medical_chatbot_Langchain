import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

from utility.utility import load_pdf, text_chunking, huggingface_embeddings, init_pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.sidebar.title("Book Selection")
book_option = st.sidebar.radio("Select source:", ["Use existing book", "Upload new book"], index=0)

vector_store = None

if book_option == "Upload new book":
   
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for f in uploaded_files:
            with open(os.path.join(temp_dir, f.name), "wb") as file:
                file.write(f.getbuffer())

        with st.spinner("Processing uploaded PDFs..."):
            docs = load_pdf(temp_dir)
            chunks = text_chunking(docs)
            embeddings = huggingface_embeddings()
            vector_store = init_pinecone(INDEX_NAME, embeddings, PINECONE_API_KEY)
            vector_store.add_documents(chunks)

        st.sidebar.success("New book added to Pinecone!")



if vector_store is None:
    with st.spinner("Initializing embeddings and Pinecone..."):
        embeddings = huggingface_embeddings()
        vector_store = init_pinecone(INDEX_NAME, embeddings, PINECONE_API_KEY)


st.title("🤖 Medical PDF Chatbot")
st.write("Ask questions based on the PDFs indexed in Pinecone.")


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

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2,google_api_key=GOOGLE_API_KEY)

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

