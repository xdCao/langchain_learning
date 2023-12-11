from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

llm = ChatOpenAI(temperature=0.9, openai_api_key=openai_api_key, openai_api_base=openai_api_base)


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


test_sequential_chain()
# test_simple_seq_chain()