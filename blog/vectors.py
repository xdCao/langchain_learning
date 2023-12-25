import configparser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

loaders = [
    # PyPDFLoader("docs/服务端开发与面试知识手册.pdf"),
    # PyPDFLoader("docs/服务端开发与面试知识手册.pdf"),
    # PyPDFLoader("docs/Go语言编程.pdf"),
    PyPDFLoader(file_path="/Users/caohao/Downloads/1_个人简历.pdf")
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

trunks = r_splitter.split_documents(documents=docs)

print(f"trunks size {len(trunks)}")

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)

vector_db = Chroma.from_documents(
    documents=trunks,
    embedding=embedding,
    persist_directory="docs/chroma/"
)

print(f"db count: {vector_db._collection.count()}")

question = "不苦的项目经历"
docs = vector_db.similarity_search(question, k=1)

print(f"result docs{len(docs)}")

print(f"result : {docs[0].page_content}")
