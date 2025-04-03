from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import time
import os

#define constants
data_folder = "./pdfs"
chunk_size = 1000
chunk_overlap = 50
check_interval = 10
DB_LOCATION = "./chroma_langchain_db"

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="all_schools",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

def ingest_file(filepath): 
    if not filepath.lower().endswith('.pdf'):
        print('Skipping non-PDF file: ', filepath)
        return

    print ('ingesting file: ', filepath)

    try:
        loader = PyPDFLoader(filepath)
        loaded_documents = loader.load()
    except Exception as e:
        print ('Error loading PDF: ', filepath)
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " ", ""]
    )

    documents = text_splitter.split_documents(loaded_documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print ('Adding ', len(documents), ' documents to the vector store')
    vector_store.add_documents(documents=documents, ids=uuids)
    print ('finished ingesting file: ', filepath)

def main_loop():
    try:
        while True:
            files_found = False
            for filename in os.listdir(data_folder):
                if not filename.startswith("_"):
                    files_found = True
                    file_path = os.path.join(data_folder, filename)
                    ingest_file(file_path)
                    new_filename = "_" + filename
                    new_file_path = os.path.join(data_folder, new_filename)
                    os.rename(file_path, new_file_path)
            if not files_found:
                print ('Now new files found. Sleeping...')
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print ('Shutting down gracefully')

if __name__ == "__main__":
    main_loop()
