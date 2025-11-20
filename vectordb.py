import os
import uuid
from typing import List ,Dict, Tuple
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from collections import defaultdict
from langchain_chroma import Chroma
import re
import logging
import numpy as np


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__) #__name__当前模块名字

def get_embedding():
    """
    获取到 embedding 实例
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDINGS_MODEL,
        model_kwargs={'device': 'cuda'},  # 或 'cuda' 如果有 GPU
        encode_kwargs={'normalize_embeddings': True}  # M3E 通常需要归一化
    )
    return embeddings


def get_chroma():
    """
    初始化 Chroma 客户端，支持持久化到本地磁盘，使用LangChain方式
    """
    vectorstore = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=get_embedding()
    )
    return vectorstore


# 4. 创建并存储到 ChromaDB
def create_vector_store(texts, persist_directory=settings.COLLECTION_NAME):
    """创建向量数据库"""
    # 方法一：直接从文档创建
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=get_embedding,
        persist_directory=persist_directory
    )

    return vectorstore



def get_similar_content(k:int = 2):
    chroma = get_chroma()
    vectorstore = chroma.as_retriever(search_kwargs={"k":k})
    return vectorstore

#计算List和Query的相似度，返回前两个相似的
def calc_relevent(texts,query,k:int = 2):
    embeddings = get_embedding()
    text_embs = embeddings.embed_documents(texts)
    query_emb = embeddings.embed_query(query)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scores = [cosine_sim(query_emb, t) for t in text_embs]
    k = 0 -k
    top2_idx = np.argsort(scores)[k:]
    return top2_idx





def clean_text_comprehensive(text):
    """全面的文本清理函数"""
    # 1. 移除前后空白
    text = text.strip()

    # 2. 合并多个空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)

    #将多个换行合成一个
    text = re.sub(r'\n+', '\n', text)

    # 3. 移除特殊空白字符（可选）
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # 4. 标准化标点符号周围的空格（可选）
    #text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
    return text

async def add_document_to_vectorstore(files):
    errors = [] #存储错误信息
    total_added_chunks = 0 #存储添加的块数

    vector_store = get_chroma() #获取向量数据库实例
    for f in files:
        try:
            #读取文件
            contents = await f.read()
            suffix = f.filename.split('.')[-1]
            tmp_path = settings.TEMP_PATH + f"/{uuid.uuid4()}.{suffix}"
            # 保存到磁盘
            with open(tmp_path, "wb") as out:
                out.write(contents)

        # 2. 准备文档数据
            """加载文档"""
            if tmp_path.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')

            #加载文档
            documents = loader.load()

            #元数据加入文件名
            for doc in documents:
                doc.metadata.clear()
                doc.metadata["source"] = f.filename

            #清洗文本
            for doc in documents:
                doc.page_content = clean_text_comprehensive(doc.page_content)

            #分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=50, separators=["。", "？"], length_function=len,
            )
            doc_splits = text_splitter.split_documents(documents)


            vector_store.add_documents(
                documents=doc_splits,
                wait=True
            )
            total_added_chunks += len(doc_splits)

            # 清理临时文件
            os.remove(tmp_path)

        except Exception as e:
            error_msg = f"失败加工文件: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue

    return total_added_chunks, errors


#把网页内容加载到数据库
def add_urls_to_vectorstore(urls: List[str]) -> Tuple[int, List[str]]:
    """
    遍历URL列表处理每个网页
    使用WebBaseLoader加载网页内容
    清理和格式化文本内容
    使用文本分割器创建文档块
    清理元数据
    将文档块添加到向量数据库
    """
    errors = []
    total_added_chunks = 0
    
    vector_store = get_chroma()

    for a_url in urls:
        try:

            docs = WebBaseLoader(a_url).load()
            docs_list = [docs] if not isinstance(docs, list) else docs

            for doc in docs_list:
                doc.page_content = clean_text_comprehensive(doc.page_content)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=50 ,separators=["。", "？"],length_function=len,
            )
            doc_splits = text_splitter.split_documents(docs_list)
            
            for doc in doc_splits:
                doc.metadata.pop("description", None)  # 安全移除 description


            vector_store.add_documents(
                documents=doc_splits, 
                wait=True
            )
            total_added_chunks += len(doc_splits)


        except Exception as e:
            error_msg = f"失败加工 URL {a_url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue  # Continue with next URL

    return total_added_chunks, errors


def get_metadata_counts() -> Dict[str, int]:
    """按元数据获取块计数"""
    vectorstore = get_chroma()
    collection = vectorstore._collection

    ## 创建一个defaultdict，默认值为
    counts = defaultdict(int)
    offset = 0
    batch_size = 300

    while True:
        # 使用get方法替代scroll
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )

        if not results['ids']:
            break

        # 处理当前批次的文档
        for metadata in results['metadatas']:
            if metadata:
                # 方式1: 直接source字段
                if 'source' in metadata:
                    source = metadata['source']
                    counts[source] += 1
                # 方式2: metadata.source字段
                elif ('metadata' in metadata and
                      isinstance(metadata['metadata'], dict) and
                      'source' in metadata['metadata']):
                    source = metadata['metadata']['source']
                    counts[source] += 1

        # 检查是否还有更多数据
        if len(results['ids']) < batch_size:
            break

        offset += batch_size

    #转换成普通字典
    return dict(counts)


def delete_by_metadata(metadata_value: str) -> int:
    """
    元数据值删除向量
    """
    # 获取底层 collection
    collection = get_chroma()._collection
    # 先统计匹配的文档数量
    count_before = collection.count()
    try:
        # 执行删除
        collection.delete(where={"source": metadata_value})
        # 删除后再次统计
        count_after = collection.count()
        deleted_count = count_before - count_after
        logger.info(f"删除完成: 删除了 {deleted_count} 个文档")
        return deleted_count

    except Exception as e:
        logger.error(f"删除失败: {e}")
        count_after = collection.count()
        deleted_count = count_before - count_after
        return deleted_count


