import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

# 打印向量数据库中的文档数量
print(vectordb._collection.count())

question = "不苦主要做了什么"
docs = vectordb.similarity_search(question, k=3)

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" \
at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

result = qa_chain({"query": question})
print(result["result"])
print(result["source_documents"])