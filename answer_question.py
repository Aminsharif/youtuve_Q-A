import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
groq_api_key = os.getenv('GROQ_API_KEY')


if len(sys.argv) < 3:
    print("Error: You must provide a vector store dir and a question as a command-line argument.")
    print("Usage: python answer_question.py <name_of_vector_store> <question>")
    sys.exit()

save_dir=sys.argv[1]
persist_directory = f'docs/chroma/{save_dir}'
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
question = sys.argv[2]

llm = ChatGroq(model="qwen/qwen3-32b", api_key=groq_api_key)

template = """ 
 Answer the question base only on provided context:
 {context}

 Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def formated_doc(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": vectordb.as_retriever()| formated_doc, "question": RunnablePassthrough() }
    | prompt
    | llm
)

result = rag_chain.invoke(question)

print(result.content) #https://youtu.be/1eym7BTnuNg