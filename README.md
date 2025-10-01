# 📄 Medical PDF Chatbot

A Streamlit-based chatbot that allows users to ask questions from medical PDFs. The app uses **LangChain** for retrieval-based QA, **HuggingFace embeddings** for semantic search, and **Pinecone** for storing vectorized document embeddings.

---

## Features

- Ask questions based on medical PDFs.
- Supports multiple questions in a chat-like interface.
- Stores PDFs in **Pinecone** for fast semantic search.
- Option to **use existing PDFs** already uploaded or **add new PDFs**.
- Chat history displayed in **reverse order** (latest question first).
- Clean and interactive **Streamlit UI**.

---

## Tech Stack

- **Frontend/UI**: Streamlit
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-l6-v2`)
- **LLM**: Google Gemini via `langchain-google-generative-ai` or OpenAI GPT-3.5 (`ChatOpenAI`)
- **Python Environment**: 3.13+

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ayushsingh991399/End_To_End_Medical_chatbot.git
cd End_To_End_Medical_chatbot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```



4. Set up environment variables:

Create a `.env` file:

```
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACE_TOKEN=your_huggingface_token  # optional if using HF API
OPENAI_API_KEY=your_openai_key            # optional if using ChatOpenAI
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. **Sidebar**:
   - Use existing PDFs in Pinecone or upload new PDFs.
   - Uploaded PDFs are processed, chunked, and embeddings stored automatically.

3. **Main Chat Interface**:
   - Type your question in the input box and click **Send**.
   - Answers appear in a **bot-like chat system**.
   - All previous questions are displayed in reverse order (latest first).

---

## Folder Structure

```
├── app.py                     # Main Streamlit app
├── utility/
│   ├── utility.py             # PDF loading, chunking, embeddings, Pinecone initialization
├── data/                      # Optional folder for default PDFs
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not committed)
└── README.md                  # This file
```

---

## Notes

- **Never commit your `.env` file** or API keys to GitHub.
- Make sure Pinecone index exists or will be created automatically by the app.
- The app works with either **Google Gemini LLM** (via `langchain-google-generative-ai`) or **OpenAI GPT-3.5**.

---

## License

MIT License

---

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

