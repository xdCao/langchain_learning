from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

chat_model = ChatOpenAI(openai_api_key=openai_api_key,
                        openai_api_base=openai_api_base)

style = """
委婉和耐心的语气表达
"""
customer_email = """
哎呀，我很生气，因为我的搅拌机盖子飞走了，冰沙溅到了我的厨房墙壁上！ 
更糟糕的是，保修不包括清理厨房的费用。 我现在需要你的帮助，朋友！
"""

template_string = """将由三个反引号分隔的文本
转换为 {style} 风格。
文本：```{text}```
"""
prompt_template = ChatPromptTemplate.from_template("请给我一段西班牙语的关于一个电子游戏商品的评论")

# customer_messages = prompt_template.format_messages(
#                     style=style,
#                     text=customer_email)

# 调用LLM来翻译客户信息的风格
customer_response = chat_model(prompt_template.format_messages())
print(customer_response.content)