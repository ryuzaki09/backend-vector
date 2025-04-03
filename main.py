from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from fastapi.responses import StreamingResponse
import asyncio
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize models and vector store
model = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_langchain_db"

vector_store = Chroma(
    collection_name="all_schools",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Define prompt template
template = """
You are an expert in answering questions about schools

Use ONLY these information about schools to answer my question: {school_info}
If you don't have the information, simply reply with "I don't have enough information to answer your query".

Here is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vector_store.as_retriever(kwargs={"k": 10})
chain = prompt | model

class QueryRequest(BaseModel):
    question: str

async def stream_response(school_info, question):
    """ Generator that streams response tokens """
    async for chunk in chain.astream({"school_info": school_info, "question": question}):
        yield f"{chunk}\n"
        await asyncio.sleep(0.1)

@app.post("/query-stream")
async def query_database_stream(request: QueryRequest):
    """ Handles AI question-answering with streaming """
    docs = retriever.invoke(request.question)
    if not docs:
        return StreamingResponse(iter(["No relevant information found."]), media_type="text/plain")

    school_info = "\n\n".join([doc.page_content for doc in docs])
    return StreamingResponse(stream_response(school_info, request.question), media_type="text/plain")


def query_vector_store():
    """Handle user interaction for querying the database"""
    while True:
        print ("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        print ("\n\n-------------------------------")
        if question == "q":
            break

        docs = retriever.invoke(question)
        school_info = "\n\n".join([doc.page_content for doc in docs])
        result = chain.invoke({"school_info": school_info, "question": question})
        print (result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name == "__main__":
    # query_vector_store()
