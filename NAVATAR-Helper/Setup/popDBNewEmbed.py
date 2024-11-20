import os
import shutil
from langchain_milvus import Milvus
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# --- FRA YOUNES SIN KODE, HENTER INN DOKUMENTENE FRA MAPPA OG LEGGER DEM TIL DB:
DATA_PATH = "./../NEET"
DATABASE_PATH = "./../Database/milvus_512_newpdf.db"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()
            
## Denne funksjonen splitter PDF'ene i chunks.
## Chunk size burde optimaliseres. Enkelte Embedding modeller har en max size på chunks, slik at alt over det blir truncated.
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
                                                        
def populate_database():
    if os.path.exists(DATABASE_PATH):
        shutil.rmtree(DATABASE_PATH)
    documents = load_documents()
    print("ANTALL DOKUMENTER: ",len(documents))
    chunks = split_documents(documents)
    
    encode_kwargs = {'prompt': 'Given a question, retrieve relevant documents that best answer the question:'}
    embedding_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct', encode_kwargs=encode_kwargs)
    
    db = Milvus.from_documents(documents=chunks,
             embedding=embedding_model,
             connection_args={"uri": DATABASE_PATH}, drop_old=True)             


populate_database()