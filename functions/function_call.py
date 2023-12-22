import json
import configparser
import pprint

import openai
from openai import OpenAI

config = configparser.ConfigParser()
config.read('../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')


def get_current_weather(location, unit="fahrenheit"):
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }
    return json.dumps(weather_info)


functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "上海的天气怎么样?"
    }
    # # {
    # #     "role": "user",
    # #     "content": "波士顿的天气怎么样?"
    # # },
    # {
    #     "role": "user",
    #     "content": "你好?"
    # }
]

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    functions=functions,
    function_call="auto",
    # function_call={"name": "get_current_weather"},  # 强制调用外部函数
)

response = json.loads(response.model_dump_json())
# 整合chatgpt的返回结果
msg = response['choices'][0]['message']
msg['tool_calls'] = []
msg['content'] = ''

# 从chatgpt的返回结果中获取外部函数的调用参数
args = json.loads(response['choices'][0]['message']['function_call']['arguments'])
# 调用外部函数
observation = get_current_weather(args)

messages.append(
    {
        "role": "function",
        "name": "get_current_weather",
        "content": observation,  # 外部函数的返回结果
    }
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
)
print(response)
