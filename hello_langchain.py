
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat_model = ChatOpenAI(openai_api_key="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")

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]
resp = chat_model.invoke(messages)
print(resp.content)
