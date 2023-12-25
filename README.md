# langchain_learning

## 学习材料
1. [入门博客](https://blog.csdn.net/weixin_42608414?type=blog)
2. 纸质书：LangChain入门指南

## 入门教程

### 秘钥设置
```python
from langchain.llms import OpenAI
import os
import getpass

# 方法1：硬编码
open_ai_key = "xxx"
llm = OpenAI(openai_api_key=open_ai_key)
# 或者通过硬编码设置环境变量, langchain会自动读取
os.environ["OPENAI_API_KEY"] = "xxxxx"
llm = OpenAI()

# 方法2：在操作系统中设置环境变量
'export OPENAI_API_KEY="xxxxxx"'

# 方法3: 使用getpass设置环境变量
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAi Api key: ")

```

### 简单取名程序
```python
from langchain.llms import OpenAI
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("输入apiKey: ")

llm = OpenAI()
response = llm.predict("我想要新建一个自媒体视频账号，用于记录自己的健身日常，请帮我取一个好名字，要求是中文")
print(response)
```

### 第一个聊天机器人
```python

```
