import re
import requests
from langchain_ollama import ChatOllama
from config import settings
from langchain_core.prompts import ChatPromptTemplate
from vectordb import get_similar_content, calc_relevent

#提示词模板
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是非常得力的助手，可以根据我提供的信息回答我的问题，返回给我文本格式，不要加符号"),
    ("human", "我们之前的聊天内容：{history} \n" + "我提供给你的信息内容：{content} \n" + "根据以上的信息回答我的问题：{question}")
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是非常得力的助手，可以将对话精炼的总结"),
    ("human", "将我们的聊天内容总结成{number}字的一段话，我们的聊天内容为：{chat_content}")
])


#聊天消息历史
history = ""

def get_history():
    global history
    return history

def llm_answer(question: str):
    global history
    
    #创建大模型
    #ollama = ChatOpenAI(model=settings.LLM_MODEL,api_key=settings.QWEN_API_KEY,base_url=settings.QWEN_BASE_URL)
    ollama = ChatOllama(model=settings.LLM_MODEL)

    #检索数据库，匹配前3个
    retriever = get_similar_content(3)
    docs = retriever.invoke(question)

    content_list = []
    metadata_list = []
    for doc in docs:
        content_list.append(doc.page_content)
        metadata_list.append(doc.metadata["source"])

    #获取网页内容 content 是数据库内容加网页内容
    web_content = baidu_search(question)
    if web_content["exists"]:
       for doc in web_content["data"]:
            content_list.append(doc["summary"])
            metadata_list.append(doc["url"])

    index_list = calc_relevent(content_list, question)


    content = "\n".join(content_list[i] for i in index_list)
    source = "\n".join(metadata_list[i] for i in index_list)
    source = "参考数据来源：\n" + source + "\n\n"
    #文档加记忆生成提示词
    end_question = answer_prompt.invoke({"question": question, "content": content ,  "history": history})

    #给大模型生成回复
    answer = ollama.invoke(end_question)


    #将回复以及记忆生成概要存储到记录中
    summary_question = summary_prompt.invoke({"number":settings.MEMORY, "chat_content":answer.content + history})
    history = ollama.invoke(summary_question).content

    ai_answer = source + "AI回复：\n" + answer.content
    clean = re.sub(r"[#$^&*]", "", ai_answer)


    #返回回复
    return clean


def baidu_search(query):
    # 请求头
    headers = {
        "Host": settings.BAIDU_HOST,
        "Authorization": f"Bearer {settings.BAIDU_API_KEY}",  # 替换为你的AppBuilder API Key
        "Content-Type": "application/json"
    }

    # 请求体（JSON数据）
    payload = {
        "messages": [
            {
                "content": query,  # 搜索关键词（此处动态传入）
                "role": "user"
            }
        ],
        "search_source": "baidu_search_v2",#搜索引擎版本
        "resource_type_filter": [
            {
                "type": "web",
                "top_k": 3  # 返回最多20条结果
            }
        ],

        "search_recency_filter": "semiyear"  # 限定搜索半年内的结果
    }

    try:
        # 发送POST请求
        response = requests.post(
            url = settings.BAIDU_API_URL,
            headers = headers,
            json = payload,  # ✅ 推荐用 json 参数自动序列化
            timeout = int(settings.TIMEOUT) # ✅ 设置超时，防止卡死
        )

        info = {
            "exists": True,
            "data":[]
        }


        # 检查请求是否成功
        if response.status_code == 200:
            data = response.json()
            for item in data["references"]:
                summary = {"summary": item["content"], "url": item["url"]}
                info["data"].append(summary.copy())

        else:
            summary = {
                "summary": [],
                "url": [],
            }
            info["exists"] = False
            info["data"].append(summary)

        return info

    except requests.exceptions.RequestException as e:

        info = {"exists" : False}
        return info





