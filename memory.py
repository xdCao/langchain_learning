from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')