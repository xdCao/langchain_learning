from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)
# 定义ConversationBufferMemory记忆力组件
# memory = ConversationBufferMemory()
# memory = ConversationBufferWindowMemory(k=1)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

conversation = ConversationChain(
    llm=llm,
    memory=memory,  # 加入ConversationBufferMemory组件
    verbose=True
)
print(conversation.predict(input="你好，我的名字叫王老六，我是一个程序员"))
print(conversation.predict(input="成功的要素有哪些"))
print(conversation.predict(input="我该怎样才能获得成功"))

