import configparser
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import requests
import datetime

config = configparser.ConfigParser()
config.read('../../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')


class SearchInput(BaseModel):
    query: str = Field(description="查询的条件")


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """查询线上天气数据"""
    return "42f"


class OpenMeteoInput(BaseModel):
    latitude: float = Field(description="Latitude of the location to fetch weather data for")
    longitude: float = Field(description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")
    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'


# pip install wikipedia

import wikipedia
from wikipedia.exceptions import PageError, DisambiguationError


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (PageError, DisambiguationError,):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


from langchain.schema.agent import AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import format_tool_to_openai_function

functions = [
    format_tool_to_openai_function(search_wikipedia),
    format_tool_to_openai_function(get_current_temperature)
]

model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key).bind(functions=functions)

from langchain.memory import ConversationBufferMemory

# 创建带有聊天历史记录变量的prompt模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建agent_chain
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

# 创建记忆力组件
memory = ConversationBufferMemory(return_messages=True,
                                  memory_key="chat_history")
# 添加记忆力组件
agent_executor = AgentExecutor(agent=agent_chain,
                               tools=[search_wikipedia, get_current_temperature],
                               verbose=True,
                               memory=memory)

# 调用chain
agent_executor.invoke({"input": "上海今天的天气怎么样"})