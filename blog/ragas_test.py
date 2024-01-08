import getpass
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable.base import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset

os.environ["OPENAI_API_KEY"] = getpass.getpass("输入apiKey: ")
# 创建BAAI的embedding
bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                                          cache_folder="../")
urls = "https://baike.baidu.com/item/恐龙/139019"

loader = WebBaseLoader(urls)
docs = loader.load()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_spliter = RecursiveCharacterTextSplitter(chunk_size=400)

vector_store = Chroma(
    collection_name="split_parents", embedding_function=bge_embeddings
)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_spliter,
    parent_splitter=parent_splitter,
    #     verbose=True,
    search_kwargs={"k": 2}
)

# 添加文档集
retriever.add_documents(docs)

model = ChatOpenAI()
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = RunnableMap(
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    }
) | prompt | model | StrOutputParser()

questions = ["恐龙是怎么被命名的？",
             "恐龙怎么分类的？",
             "体型最大的是哪种恐龙?",
             "体型最长的是哪种恐龙？它在哪里被发现？",
             "恐龙采样什么样的方式繁殖？",
             "恐龙是冷血动物吗？",
             "陨石撞击是导致恐龙灭绝的原因吗？",
             "恐龙是在什么时候灭绝的？",
             "鳄鱼是恐龙的近亲吗？",
             "恐龙在英语中叫什么？"
             ]
ground_truths = [
    [
        "1841年，英国科学家理查德·欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙，意思是“恐怖的蜥蜴”。"
    ],
    ["恐龙可分为鸟类和非鸟恐龙。"],
    ["恐龙整体而言的体型很大。以恐龙作为标准来看，蜥脚下目是其中的巨无霸。"],
    ["最长的恐龙是27米长的梁龙，是在1907年发现于美国怀俄明州。"],
    ["恐龙采样产卵、孵蛋的方式繁殖。"],
    ["恐龙是介于冷血和温血之间的动物"],
    [
        "科学家最新研究显示，0.65亿年前小行星碰撞地球时间或早或晚都可能不会导致恐龙灭绝，真实灭绝原因是当时恐龙处于较脆弱的生态系统中，环境剧变易导致灭绝。"],
    ["恐龙灭绝的时间是在距今约6500万年前，地质年代为中生代白垩纪末或新生代第三纪初。"],
    ["鳄鱼是另一群恐龙的现代近亲，但两者关系较非鸟恐龙与鸟类远。"],
    [
        "1842年，英国古生物学家理查德·欧文创建了“dinosaur”这一名词。英文的dinosaur来自希腊文deinos（恐怖的）Saurosc（蜥蜴或爬行动物）。对当时的欧文来说，这“恐怖的蜥蜴”或“恐怖的爬行动物”是指大的灭绝的爬行动物（实则不是）"
    ]
]
answers = []
contexts = []

# Inference
for query in questions:
    answers.append(chain.invoke({"question": query}))
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
result = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
)
df = result.to_pandas()

print(df)
