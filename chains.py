from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain, LLMRouterChain, MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import RouterOutputParser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)


def test_simple_seq_chain():
    name_prompt = PromptTemplate.from_template("描述生产{product}的公司的一个最佳名称是什么？")
    name_chain = LLMChain(llm=llm, prompt=name_prompt)
    desc_prompt = PromptTemplate.from_template("为以下公司编写 20 个字的描述：{company_name}")
    desc_chain = LLMChain(llm=llm, prompt=desc_prompt)
    seq_chain = SimpleSequentialChain(chains=[name_chain, desc_chain], verbose=True)
    output = seq_chain.run("枕头")
    print(output)


def test_sequential_chain():
    tr_ch_prompt = PromptTemplate.from_template(
        "将下面的评论翻译成中文:"
        "\n\n{Review}"
    )
    chain_1 = LLMChain(llm=llm, prompt=tr_ch_prompt, output_key="Chinese_Review")
    sum_prompt = PromptTemplate.from_template(
        "你能用 1 句话概括以下评论吗："
        "\n\n{Chinese_Review}"
    )
    chain_2 = LLMChain(llm=llm, prompt=sum_prompt, output_key="summary")

    reg_lang_prompt = PromptTemplate.from_template("这段话：{Review}\n\n使用的是什么语言")
    chain_3 = LLMChain(llm=llm, prompt=reg_lang_prompt, output_key="language")

    reply_prompt = PromptTemplate.from_template("使用指定语言编写对以下摘要的后续回复,摘要:{summary},语言:{language}")
    chain_4 = LLMChain(llm=llm, prompt=reply_prompt, output_key="reply")

    five_prompt = PromptTemplate.from_template(
        "将下面的评论翻译成中文:"
        "\n\n{reply}"
    )
    chain_5 = LLMChain(llm=llm, prompt=five_prompt, output_key="result")

    overall_chain = SequentialChain(
        chains=[chain_1, chain_2, chain_3, chain_4, chain_5],
        input_variables=["Review"],
        output_variables=["language", "Chinese_Review", "summary",
                          "reply", "result"],
        verbose=True
    )
    print(overall_chain(
        "Este juego es increíble. La calidad gráfica es impresionante y la jugabilidad es muy adictiva. Me encanta la "
        "variedad de personajes y niveles que ofrece. Además, el modo multijugador es genial para jugar con amigos. "
        "El único inconveniente es que a veces los servidores pueden ser un poco lentos, pero en general, "
        "es un producto excelente. ¡Definitivamente lo recomendaría a todos los amantes de los videojuegos!"))


def test_router_chain():
    physics_template = """你是一位非常聪明的物理学教授。\
    你擅长以简洁易懂的方式回答有关物理的问题。 \
    当你不知道某个问题的答案时，你就承认你不知道。
    这里有一个问题：
    {input}"""

    math_template = """你是一位非常优秀的数学家。\
    你很擅长回答数学问题。 \
    你之所以如此出色，是因为你能够将难题分解为各个组成部分，\
    回答各个组成部分，然后将它们组合起来回答更广泛的问题。
    这里有一个问题：
    {input}"""

    history_template = """你是一位非常优秀的历史学家。\
    你对各个历史时期的人物、事件和背景有深入的了解和理解。 \
    你有能力思考、反思、辩论、讨论和评价过去。 \
    你尊重历史证据，并有能力利用它来支持你的解释和判断。
    这里有一个问题：
    {input}"""

    computerscience_template = """你是一位成功的计算机科学家。\
    你有创造力，协作精神，前瞻性思维，自信，有很强的解决问题的能力，\
    对理论和算法的理解，以及出色的沟通能力。\
    你很擅长回答编程问题。
    你是如此优秀，因为你知道如何通过描述一个机器可以很容易理解的命令步骤来解决问题，\
    你知道如何选择一个解决方案，在时间复杂度和空间复杂度之间取得良好的平衡。
    这里有一个问题：
    {input}"""

    prompt_infos = [
        {
            "name": "physics",
            "description": "擅长回答有关物理方面的问题",
            "prompt_template": physics_template
        },
        {
            "name": "math",
            "description": "擅长回答有关数学方面的问题",
            "prompt_template": math_template
        },
        {
            "name": "history",
            "description": "擅长回答有关历史方面的问题",
            "prompt_template": history_template
        },
        {
            "name": "computer science",
            "description": "擅长回答有关计算机科学方面的问题",
            "prompt_template": computerscience_template
        }
    ]

    destination_chains = {}
    for prompt_info in prompt_infos:
        name = prompt_info["name"]
        prompt_template = prompt_info["prompt_template"]
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    print(destinations_str)

    default_prompt = PromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    # MULTI_PROMPT_ROUTER_TEMPLATE = """给定一个原始文本输入到\
    # 一个语言模型并且选择最适合输入的模型提示语。＼
    # 你将获得可用的提示语的名称以及该提示语最合适的描述。＼
    # 如果你认为修改原始输入最终会导致语言模型得到更好的响应，你也可以修改原始输入。
    # << FORMATTING >>
    # 返回一个 Markdown 代码片段，其中 JSON 对象的格式如下：
    # ```json
    # {{{{
    # "destination": string \ 要使用的提示语的名称或"DEFAULT"
    # "next_inputs": string \ 原始输入的可能修改版本
    # }}}}
    # ```
    # 记住:"destination"必须是下面指定的候选提示语中的一种，\
    # 如果输入语句不适合任何候选提示语，则它就是"DEFAULT"。
    # 记住:"next_inputs"可以只是原始输入，如果你认为不需要做任何修改的话。
    # << CANDIDATE PROMPTS >>
    # {destinations}
    # << INPUT >>
    # {{input}}
    # << OUTPUT (remember to include the ```json)>>"""
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    print(router_prompt)

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(router_chain=router_chain,
                             destination_chains=destination_chains,
                             default_chain=default_chain, verbose=True
                             )
    response = chain.run("成吉思汗是谁？")
    print(response)


test_router_chain()
# test_sequential_chain()
# test_simple_seq_chain()