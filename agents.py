import configparser
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import tool, DuckDuckGoSearchRun, Tool
from datetime import date

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)

duck = DuckDuckGoSearchRun()
duck_tool = Tool(
    name="duck",
    func=duck.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. \
    be specific with your input."
)

tools = load_tools(["llm-math", "wikipedia"], llm=llm)


# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                          handle_parsing_errors=True,
#                          verbose=True)
# print(agent("300的25%是多少？"))
# print(agent("西游记的作者是谁？请用中文回答我"))

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


agent = initialize_agent(
    tools + [time, duck_tool],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True)

try:
    result = agent("昨天阿里巴巴股价开盘是多少")
    print(result)
except:
    print("exception on external access")
