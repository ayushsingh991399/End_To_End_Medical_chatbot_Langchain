from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone
from langchain_pinecone import PineconeVectorStore

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents


def text_chunking(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 30)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


def huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    )
    return embeddings



def init_pinecone(index_name, embeddings, api_key, environment="us-east-1"):
    # Modern Pinecone initialization
    pinecone.init(api_key=api_key, environment=environment)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
        )

    index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store



