# 构建 RAG 增强检索生成实战



前提说明：访问网络中的大模型都是按 token 计费的，所以我退而求其次，用了本地的模型，如果想用网络大模型很简单，只需要更改个参数，加入计费的 API_KEY 即可，文中对应部分会提及。记忆上下文我默认设置了 400 ，如果想要记忆的更多更准，只需要把数字调大即可。





## 搭建环境变量

> 1. **首先创建一个 .env 文件，用于配置环境变量**
> 2. **这个项目需要配置的环境如下：**
>
> ```python
> #向量数据库名称
> COLLECTION_NAME=chroma.db
> #嵌入模型
> EMBEDDINGS_MODEL=F:\models\m3e\m3e-base
> #LLM 模型(本地模型）
> LLM_MODEL=qwen3:1.7b
> #千问模型的 API_KEY（阿里大模型）
> QWEN_API_KEY=
> #千问模型的地址 URL
> QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
> #向量数据地址
> CHROMA_PATH=F:\models\chrome_db\chroma.db
> #百度搜索主机地址
> BAIDU_HOST=qianfan.baidubce.com
> #百度搜索 API_URL
> BAIDU_API_URL=https://qianfan.baidubce.com/v2/ai_search/web_search
> #百度搜索 API_KEY
> BAIDU_API_KEY=
> #网络请求超时时间
> TIMEOUT=30
> #临时目录地址
> TEMP_PATH=.\tmp
> #短期记忆存储的长度，如果想要记忆上下文更多的话，把数字改大
> MEMORY=400
> ```
>
> 
>
> 3. **创建一个 config.py 文件，目的是方便环境配置的调用**
>
> ```python
> import os
> from dataclasses import dataclass
> import dotenv
> 
> #加载环境变量
> dotenv.load_dotenv()
> @dataclass
> class Settings:
>     COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "chroma.db")
>     EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-mpnet-base-v2")
>     QWEN_API_KEY:str = os.getenv("QWEN_API_KEY", "")
>     QWEN_BASE_URL:str = os.getenv("QWEN_BASE_URL", "")
>     CHROMA_PATH: str = os.getenv("CHROMA_PATH", "")
>     BAIDU_API_URL: str = os.getenv("BAIDU_API_URL", "")
>     BAIDU_API_KEY: str = os.getenv("BAIDU_API_KEY", "")
>     TIMEOUT: int = os.getenv("TIMEOUT", "")
>     BAIDU_HOST: str = os.getenv("BAIDU_HOST", "")
>     TEMP_PATH: str = os.getenv("TEMP_PATH", "")
>     LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")
>     MEMORY: int = os.getenv("MEMORY", "")
> 
> settings = Settings()
> ```
>
> 



## 向量数据库

**我喜欢从低端向上去构建整个系统，首先是先有个数据库用于检索数据，综合我的电脑配置，我选择了 chromadb 数据库**

> 1. **首先导入此部分需要用到的 Python 包**
>
> ```python
> import os
> import uuid
> from typing import List ,Dict, Tuple
> from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
> from langchain_huggingface import HuggingFaceEmbeddings
> from langchain_text_splitters import RecursiveCharacterTextSplitter
> from config import settings
> from collections import defaultdict
> from langchain_chroma import Chroma
> import re
> import logging
> import numpy as np
> 
> #这里配置一下日志，只显示错误及以上级别的信息，方便自己改 BUG
> logging.basicConfig(level=logging.ERROR)
> logger = logging.getLogger(__name__) #__name__当前模块名字
> ```
>
> 
>
> 2. **我封装了一个函数，用于获取 Chroma 实例**
>
> ```python
> def get_chroma():
>     """
>     初始化 Chroma 客户端，支持持久化到本地磁盘，使用LangChain方式
>     """
>     vectorstore = Chroma(
>         persist_directory=settings.CHROMA_PATH,
>         embedding_function=get_embedding()
>     )
>     return vectorstore
> ```
>
> 
>
> 3. **封装一个 Embedding 函数，获取 Embedding 实例**
>
> ```python
> def get_embedding():
>     """
>     获取到 embedding 实例
>     """
>     embeddings = HuggingFaceEmbeddings(
>         model_name=settings.EMBEDDINGS_MODEL,
>         model_kwargs={'device': 'cuda'},  # 或 'cuda' 如果有 GPU
>         encode_kwargs={'normalize_embeddings': True}  # M3E 通常需要归一化
>     )
>     return embeddings
> 
> ```
>
> 
>
> 4. **创建数据库，并可以存储数据**
>
>     将传入的文本内容持久化存储到指定向量数据库
>
> ```python
> def create_vector_store(texts, persist_directory=settings.COLLECTION_NAME):
>     """创建向量数据库"""
>     vectorstore = Chroma.from_documents(
>         documents=texts,
>         embedding=get_embedding,
>         persist_directory=persist_directory
>     )
> 
>     return vectorstore
> ```
>
> 
>
> 5. **清洗文本数据**
>
>     **在存入数据库之前，要将数据进行清洗**
>
>     ```Python
>     def clean_text_comprehensive(text):
>         """全面的文本清理函数"""
>         # 1. 移除前后空白
>         text = text.strip()
>     
>         # 2. 合并多个空白字符为单个空格
>         text = re.sub(r'\s+', ' ', text)
>     
>         #将多个换行合成一个
>         text = re.sub(r'\n+', '\n', text)
>     
>         # 3. 移除特殊空白字符（可选）
>         text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
>     
>         # 4. 标准化标点符号周围的空格（可选）
>         #text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
>         return text
>     ```
>
>     
>
> 6. **将 文本 以及 PDF 数据存入数据库**
>
> ```python
> async def add_document_to_vectorstore(files):
>     errors = [] #存储错误信息
>     total_added_chunks = 0 #存储添加的块数
> 
>     vector_store = get_chroma() #获取向量数据库实例
>     for f in files:
>         try:
>             #读取文件
>             contents = await f.read()
>             suffix = f.filename.split('.')[-1]
> 
>             tmp_path = settings.TEMP_PATH + f"/{uuid.uuid4()}.{suffix}"
>             # 保存到磁盘
>             with open(tmp_path, "wb") as out:
>                 out.write(contents)
> 
>         # 2. 准备文档数据
>             """加载文档"""
>             if tmp_path.endswith('.pdf'):
>                 loader = PyPDFLoader(tmp_path)
>             else:
>                 loader = TextLoader(tmp_path, encoding='utf-8')
> 
>             #加载文档
>             documents = loader.load()
> 
>             #元数据加入文件名
>             for doc in documents:
>                 doc.metadata.clear()
>                 doc.metadata["source"] = f.filename
> 
>             #清洗文本
>             for doc in documents:
>                 doc.page_content = clean_text_comprehensive(doc.page_content)
> 
>             #分割文本
>             text_splitter = RecursiveCharacterTextSplitter(
>                 chunk_size=300, chunk_overlap=50, separators=["。", "？"], length_function=len,
>             )
>             doc_splits = text_splitter.split_documents(documents)
> 
> 
>             vector_store.add_documents(
>                 documents=doc_splits,
>                 wait=True
>             )
>             total_added_chunks += len(doc_splits)
> 
>             # 清理临时文件
>             os.remove(tmp_path)
> 
>         except Exception as e:
>             error_msg = f"失败加工文件: {str(e)}"
>             logger.error(error_msg)
>             errors.append(error_msg)
>             continue
> 
>     return total_added_chunks, errors
> ```
>
>  
>
> 7. **将网页内容加载到向量数据库**
>
> ```python
> ef add_urls_to_vectorstore(urls: List[str]) -> Tuple[int, List[str]]:
>     """
>     遍历URL列表处理每个网页
>     使用WebBaseLoader加载网页内容
>     清理和格式化文本内容
>     使用文本分割器创建文档块
>     清理元数据
>     将文档块添加到向量数据库
>     """
>     errors = []
>     total_added_chunks = 0
>     
>     vector_store = get_chroma()
> 
>     for a_url in urls:
>         try:
> 
>             docs = WebBaseLoader(a_url).load()
>             docs_list = [docs] if not isinstance(docs, list) else docs
> 
>             for doc in docs_list:
>                 doc.page_content = clean_text_comprehensive(doc.page_content)
> 
>             text_splitter = RecursiveCharacterTextSplitter(
>                 chunk_size=300, chunk_overlap=50 ,separators=["。", "？"],length_function=len,
>             )
>             doc_splits = text_splitter.split_documents(docs_list)
>             
>             for doc in doc_splits:
>                 doc.metadata.pop("description", None)  # 安全移除 description
> 
> 
>             vector_store.add_documents(
>                 documents=doc_splits, 
>                 wait=True
>             )
>             total_added_chunks += len(doc_splits)
> 
> 
>         except Exception as e:
>             error_msg = f"失败加工 URL {a_url}: {str(e)}"
>             logger.error(error_msg)
>             errors.append(error_msg)
>             continue  # Continue with next URL
> 
>     return total_added_chunks, errors
> ```
>
>  
>
> 8. **获取向量库中元数据以及对应的块数**
>
> ```python
> def get_metadata_counts() -> Dict[str, int]:
>     """按元数据获取块计数"""
>     vectorstore = get_chroma()
>     collection = vectorstore._collection
> 
>     ## 创建一个defaultdict，默认值为
>     counts = defaultdict(int)
>     offset = 0 #分页查询的第一个索引
>     batch_size = 300
> 
>     while True:
>         results = collection.get(
>             limit=batch_size,
>             offset=offset,
>             include=["metadatas"]
>         )
> 
>         if not results['ids']:
>             break
> 
>         # 处理当前批次的文档
>         for metadata in results['metadatas']:
>             if metadata:
>                 # 方式1: 直接source字段
>                 if 'source' in metadata:
>                     source = metadata['source']
>                     counts[source] += 1
>                 # 方式2: metadata.source字段
>                 elif ('metadata' in metadata and
>                       isinstance(metadata['metadata'], dict) and
>                       'source' in metadata['metadata']):
>                     source = metadata['metadata']['source']
>                     counts[source] += 1
> 
>         # 检查是否还有更多数据
>         if len(results['ids']) < batch_size:
>             break
> 
>         offset += batch_size
> 
>     #转换成普通字典
>     return dict(counts)
> ```
>
> 
>
> 9. **根据向量库中的元数据删除相对应块数据**
>
> ```python
> def delete_by_metadata(metadata_value: str) -> int:
>     """
>     元数据值删除向量
>     """
>     # 获取底层 collection
>     collection = get_chroma()._collection
>     # 先统计匹配的文档数量
>     count_before = collection.count()
>     try:
>         # 执行删除
>         collection.delete(where={"source": metadata_value})
>         # 删除后再次统计
>         count_after = collection.count()
>         deleted_count = count_before - count_after
>         logger.info(f"删除完成: 删除了 {deleted_count} 个文档")
>         return deleted_count
> 
>     except Exception as e:
>         logger.error(f"删除失败: {e}")
>         count_after = collection.count()
>         deleted_count = count_before - count_after
>         return deleted_count
> ```
>
> 
>
> 10. **从数据库中返回和输入内容相似的前 k 条数据**
>
> ```python
> def get_similar_content(k:int = 2):
>     chroma = get_chroma()
>     vectorstore = chroma.as_retriever(search_kwargs={"k":k})
>     return vectorstore
> ```
>
> 
>
> 11. **计算传入数据的相似性，返回前 k 个相似的**
>
> ```python
> def calc_relevent(texts,query,k:int = 2):
>     embeddings = get_embedding()
>     text_embs = embeddings.embed_documents(texts)
>     query_emb = embeddings.embed_query(query)
> 
>     def cosine_sim(a, b):#余弦相似度公式
>         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
> 
>     scores = [cosine_sim(query_emb, t) for t in text_embs]
>     k = 0 -k
>     top2_idx = np.argsort(scores)[k:]
>     return top2_idx
> ```
>
> 



## 访问大模型

**数据库基本操作搭建完了，可以着手构建大模型的访问逻辑了**

> 1. **需要导入的包**
>
> ```python
> import re
> import requests
> from langchain_ollama import ChatOllama
> from config import settings
> from langchain_core.prompts import ChatPromptTemplate
> from vectordb import get_similar_content, calc_relevent
> ```
>
> 
>
> 2. **提问大模型的时候用到的提示词模板**
>
> ```python
> #用问题时候的提示词
> answer_prompt = ChatPromptTemplate.from_messages([
>     ("system", "你是非常得力的助手，可以根据我提供的信息回答我的问题，返回给我文本格式，不要加符号"),
>     ("human", "我们之前的聊天内容：{history} \n" + "我提供给你的信息内容：{content} \n" + "根据以上的信息回答我的问题：{question}")
> ])
> 
> #对之前的问题内容形成总结，
> summary_prompt = ChatPromptTemplate.from_messages([
>     ("system", "你是非常得力的助手，可以将对话精炼的总结"),
>     ("human", "将我们的聊天内容总结成{number}字的一段话，我们的聊天内容为：{chat_content}")
> ])
> 
> #聊天消息历史
> history = ""
> 
> ```
>
> 
>
> 3. **返回历史数据给前端**
>
> ```python
> def get_history():
>     global history
>     return history
> ```
>
> 
>
> 4. **调用百度搜索**
>
> ```python
> def baidu_search(query):
>     # 请求头
>     headers = {
>         "Host": settings.BAIDU_HOST,
>         "Authorization": f"Bearer {settings.BAIDU_API_KEY}",  
>         "Content-Type": "application/json"
>     }
> 
>     # 请求体（JSON数据）
>     payload = {
>         "messages": [
>             {
>                 "content": query,  # 搜索关键词（此处动态传入）
>                 "role": "user"
>             }
>         ],
>         "search_source": "baidu_search_v2",#搜索引擎版本
>         "resource_type_filter": [
>             {
>                 "type": "web",
>                 "top_k": 3  # 返回最多20条结果
>             }
>         ],
> 
>         "search_recency_filter": "semiyear"  # 限定搜索半年内的结果
>     }
> 
>     try:
>         # 发送POST请求
>         response = requests.post(
>             url = settings.BAIDU_API_URL,
>             headers = headers,
>             json = payload,  # ✅ 推荐用 json 参数自动序列化
>             timeout = int(settings.TIMEOUT) # ✅ 设置超时，防止卡死
>         )
> 
>         info = {
>             "exists": True,
>             "data":[]
>         }
> 
> 
>         # 检查请求是否成功
>         if response.status_code == 200:
>             data = response.json()
>             for item in data["references"]:
>                 summary = {"summary": item["content"], "url": item["url"]}
>                 info["data"].append(summary.copy())
> 
>         else:
>             summary = {
>                 "summary": [],
>                 "url": [],
>             }
>             info["exists"] = False
>             info["data"].append(summary)
> 
>         return info
> 
>     except requests.exceptions.RequestException as e:
> 
>         info = {"exists" : False}
>         return info
> 
> ```
>
> 
>
> 4. **调用模型**
>
> **调用过程函数逻辑：**
>
> 传入的参数(用户问题) -> 根据问题检索本地的向量数据库 -> 根据问题检索互联网内容  -> 拼合两部分内容  -> 从拼合内容中返回前 k（自己指定）条最相近的内容 -> 根据内容和聊天历史生成提示词 -> 将提示词发送给大模型得到回复1 -> 将大模型回复1内容以及之前的历史合并 -> 将合并内容生成总结提示词 -> 将提示词发送给大模型总结得到回复2 ->  将回复2存入历史记录 -> 将回复1内容返回给用户
>
> ```Python
> def llm_answer(question: str):
>     global history
>     
>     #创建大模型
>     #ollama = ChatOpenAI(model=settings.LLM_MODEL,api_key=settings.QWEN_API_KEY,base_url=settings.QWEN_BASE_URL)
>     ollama = ChatOllama(model=settings.LLM_MODEL)
> 
>     #检索数据库，匹配前3个
>     retriever = get_similar_content(3)
>     docs = retriever.invoke(question)
> 
>     content_list = []
>     metadata_list = []
>     for doc in docs:
>         content_list.append(doc.page_content)
>         metadata_list.append(doc.metadata["source"])
> 
>     #获取网页内容 content 是数据库内容加网页内容
>     web_content = baidu_search(question)
>     if web_content["exists"]:
>        for doc in web_content["data"]:
>             content_list.append(doc["summary"])
>             metadata_list.append(doc["url"])
> 
>     index_list = calc_relevent(content_list, question)
> 
> 
>     content = "\n".join(content_list[i] for i in index_list)
>     source = "\n".join(metadata_list[i] for i in index_list)
>     source = "参考数据来源：\n" + source + "\n\n"
>     #文档加记忆生成提示词
>     end_question = answer_prompt.invoke({"question": question, "content": content ,  "history": history})
> 
>     #给大模型生成回复
>     answer = ollama.invoke(end_question)
> 
> 
>     #将回复以及记忆生成概要存储到记录中
>     summary_question = summary_prompt.invoke({"number":settings.MEMORY, "chat_content":answer.content + history})
>     history = ollama.invoke(summary_question).content
> 
>     ai_answer = source + "AI回复：\n" + answer.content
>     clean = re.sub(r"[#$^&*]", "", ai_answer)
> 
> 
>     #返回回复
>     return clean
> ```
>
> 



## 构建返回类

**将内容打包返回给前端**

> ```python
> '''
> pydantic.BaseModel 可以定义一个“数据模型”，在 FastAPI 里：
> 自动验证请求数据格式是否正确；
> 自动生成文档（Swagger）；
> 自动把 JSON 转换为 Python 对象。
> '''
> 
> from pydantic import BaseModel
> from typing import List, Dict
> class History(BaseModel):
>     history : str
> 
> 
> #接受前端问题
> class QuestionRequest(BaseModel):
>     question: str
> 
> #返回回答内容和耗时
> class AnswerResponse(BaseModel):
>     answer: str
>     processing_time: str
> 
> #用于批量注入外部网页或知识来源（典型 RAG 操作）
> class UrlInjectionRequest(BaseModel):
>     urls: List[str]
> 
> #删除某个已存入向量数据库的文档。
> class DeleteRequest(BaseModel):
>     url: str
> 
> #返回向量数据库中按某种标签统计的元数据数量。
> class MetadataQueryResponse(BaseModel):
>     metadata_counts: Dict[str, int]
> 
> '''
> 用于调试向量数据库（如 Qdrant、Chroma、Milvus）的数据点。
> 一个向量点（DebugPoint）包括：
> id: 唯一标识符；
> payload: 元数据；
> vector: 实际的向量。
> '''
> class DebugPoint(BaseModel):
>     id: str
>     payload: Dict #payload: 元数据；
>     vector: List[float]
> 
> class DebugResponse(BaseModel):
>     points: List[DebugPoint]    
> 
> 
> ```
>
> 



## 前端调用接口

> 1. **导入所需要的包**
>
> ```python
> from typing import List
> from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
> from fastapi.middleware.cors import CORSMiddleware
> from contextlib import asynccontextmanager
> from llm_service.mychain import llm_answer, get_history
> from schemas import QuestionRequest, AnswerResponse, UrlInjectionRequest, MetadataQueryResponse, DeleteRequest
> from vectordb import add_urls_to_vectorstore, get_metadata_counts, delete_by_metadata, get_chroma ,add_document_to_vectorstore
> from config import settings
> import os
> import time
> import logging
> import sys
> ```
>
> 
>
> 2. **配置日志以及一些变量**
>
> ```python
> #设置成FALSE，防止PostHog 在没网的时候无限次请求
> os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
> os.environ["ANONYMIZED_TELEMETRY4"] = "False"
> 
> 
> LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() #设置日志
> logging.basicConfig( #这行是配置全局日志系统的输出格式和最低等级
>     level=LOG_LEVEL,
>     format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
>     handlers=[
>         logging.StreamHandler(sys.stdout),   # 输出到控制台
>     ],
>     force=True# ✅ 防止被 uvicorn 默认配置覆盖
> )
> logger = logging.getLogger("RAG_API")#创建一个名为 "agentic_rag_api" 的日志记录器；你后面可以用它记录项目内各种运行信息。
> 
> @asynccontextmanager
> async def lifespan(app: FastAPI): #这段是 FastAPI 的应用生命周期管理函数（lifespan）
>     # Startup logic
>     logger.info("RAG 服务开始启动-------------")
>     yield  # App runs here  FastAPI 在这里进入“正常运行状态”，处理请求。
> 
>     logger.info("关闭 RAG API.")
> 
> app = FastAPI(title="Agentic RAG API" , lifespan=lifespan)
> 
> app.add_middleware(
>     CORSMiddleware,
>     #allow_origins=["http://localhost:80"],  #  # 前端页面的来源
>     allow_origins=["*"],                  #  放开所有来源（调试时可用）
>     allow_credentials=True,# 允许携带 Cookie / Authorization 头
>     allow_methods=["*"],# 允许所有 HTTP 方法（GET/POST/PUT/DELETE...）
>     allow_headers=["*"],# 允许所有自定义 Header
> )
> 
> 
> ```
>
> 
>
> 3. **上传文本以及 PDF 文件接口**
>
> ```python
> @app.post("/uploadpdf")
> async def upload_pdf(files: List[UploadFile] = File(...)):
> 
>     try:
>         # files 是一个列表，里面每个都是 UploadFile 对象
>         added_count, errors = await add_document_to_vectorstore(files)
> 
>         if errors:
>             return {
>                 "message": f"添加 {added_count} 个块出现 {len(errors)} 个错误",
>                 "errors": errors,
>                 "status": "partial_success",
>                 "added_count": added_count,
>             }
> 
>         return {
>             "message": f"成功添加 {added_count} 个块",
>             "status": "success",
>             "added_count": added_count,
>         }
> 
>         return {"message": f"成功上传 {len(files)} 个文件"}
> 
>     except Exception as e:
>         logger.exception("失败在接口 /uploadpdf: %s", e)
>         raise HTTPException(status_code=500, detail="文件服务错误")
> 
> 
> ```
>
> 
>
> 4. **问题请求接口**
>
> ```python
> @app.post("/ask", response_model=AnswerResponse)
> async def ask_question(request: QuestionRequest):
>     """
>     向 Agentic RAG 管道提问。
>     返回漂亮打印的执行跟踪作为透明度的答案。
>     """
>     start = time.perf_counter()
>     try:
>         answer_text = llm_answer(request.question)
> 
>         elapsed = time.perf_counter() - start#计算耗时
>         return AnswerResponse(answer=answer_text, processing_time=f"{elapsed:.2f}")
>     except Exception as e:
>         logger.exception("失败在接口 /ask: %s", e)
>         raise HTTPException(status_code=500, detail="服务错误")
> 
> ```
>
> 
>
> 5. **获取历史记录接口**
>
> ```python
> @app.get("/history")
> async def history():
>     return get_history()
> ```
>
> 
>
> 6. **插入网页内容接口**
>
> ```Python
> @app.post("/inject") #向数据库插入数据
> async def inject_urls(request: UrlInjectionRequest, background_tasks: BackgroundTasks):
>     """
>     将URL注入向量存储，然后在后台安排检索器刷新。如果部分URL失败，返回部分成功信息。
>     """
>     try:
>         if not request.urls:
>             raise HTTPException(status_code=400, detail="没有提交 URL")
> 
>         added_count, errors = add_urls_to_vectorstore(request.urls)
> 
> 
>         if errors:
>             return {
>                 "message": f"添加 {added_count} 个块出现 {len(errors)} 个错误",
>                 "errors": errors,
>                 "status": "partial_success",
>                 "added_count": added_count,
>             }
> 
>         return {
>             "message": f"成功添加 {added_count} 个块",
>             "status": "success",
>             "added_count": added_count,
>         }
>     except HTTPException:
>         raise
>     except Exception as e:
>         logger.exception("意外错误在 /inject: %s", e)
>         raise HTTPException(status_code=500, detail="网络意外错误")
> ```
>
> 
>
> 7. **根据元数据删除内容信息接口**
>
> ```python
> @app.post("/delete_by_metadata")
> async def delete_by_metadata_endpoint(request: DeleteRequest, background_tasks: BackgroundTasks):
>     """
>     按元数据值（例如 URL）删除向量。
>     始终在后台安排检索器刷新。
>     """
>     try:
>         if not request.url:
>             raise HTTPException(status_code=400, detail="确实 'url' 请求")
> 
>         deleted_count = delete_by_metadata(request.url)
> 
> 
>         if deleted_count > 0:
>             return {
>                 "message": f"Successfully deleted {deleted_count} chunks with metadata '{request.url}'",
>                 "deleted_count": deleted_count,
>                 "status": "success",
>             }
>         else:
>             return {
>                 "message": f"No chunks found with metadata '{request.url}'",
>                 "deleted_count": 0,
>                 "status": "no_match",
>             }
>     except HTTPException:
>         raise
>     except Exception as e:
>         logger.exception("错误在接口 /delete_by_metadata: %s", e)
>         raise HTTPException(status_code=500, detail="网络意外错误")
> ```
>
> 
>
> 8. **获取元数据以及对应数量接口**
>
> ```Python
> @app.get("/metadata/counts", response_model=MetadataQueryResponse)
> async def get_metadata_counts_endpoint():
>     """按元数据获取块计数"""
>     try:
>         counts = get_metadata_counts()
>         return MetadataQueryResponse(metadata_counts=counts)
>     except Exception as e:
>         raise HTTPException(status_code=500, detail=str(e))
> ```
>
> 
>
> 9. **获取向量数据库内容接口**
>
> ```python
> @app.get("/debug/points")
> async def debug_points(limit: int = 50):
>     """调试终结点以查看数据库中存储的内容"""
>     try:
>         vectorstore = get_chroma()
>         collection = vectorstore._collection
> 
>         results = collection.get(
>             limit=limit,
>             include=["metadatas","documents"]  # 包含元数据和文档内容
>         )
> 
> 
>         debug_text = ""
>         for i, doc_id in enumerate(results['ids']):
>             source = results['metadatas'][i].get('source', '') if i < len(results['metadatas']) else ''
>             content_preview = results['documents'][i][:200] + "..." if i < len(results['documents']) else ""
> 
>             debug_text += f"数据源：{source}\n内容：{content_preview}\n\n"
> 
>         return debug_text
>     except Exception as e:
>         raise HTTPException(status_code=500, detail=str(e))
>     
> ```
>
> 
>
> 10. **获取配置信息接口**
>
> ```Python
> @app.get("/api/config")
> async def get_config():
>     """
>     加载配置文件在前端显示
>     """
>     ss = settings.EMBEDDINGS_MODEL
>     return {
>         "llm_model": settings.LLM_MODEL,
>         "embeddings_model": ss.rsplit("\\", 1)[-1]#将字符串分割，取右面
>     }
> ```
>
> 
>
> 11**.启动入口**
>
> ```Python
> if __name__ == "__main__":
>     import uvicorn
>     uvicorn.run(app, host="localhost", port=8001)
> ```
>
> 



**后端到这里就搭建完成了，前端可以自己去 github 或者 gitee 上找个前端，改吧改吧就可以跑了**

获取前后端源码关注微信公众号 **星辰微语阁**  回复 **RAG增强检索生成实战项目**







