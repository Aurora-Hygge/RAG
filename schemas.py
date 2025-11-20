'''
pydantic.BaseModel 可以定义一个“数据模型”，在 FastAPI 里：
自动验证请求数据格式是否正确；
自动生成文档（Swagger）；
自动把 JSON 转换为 Python 对象。
'''

from pydantic import BaseModel
from typing import List, Dict
class History(BaseModel):
    history : str


#接受前端问题
class QuestionRequest(BaseModel):
    question: str

#返回回答内容和耗时
class AnswerResponse(BaseModel):
    answer: str
    processing_time: str

#用于批量注入外部网页或知识来源（典型 RAG 操作）
class UrlInjectionRequest(BaseModel):
    urls: List[str]

#删除某个已存入向量数据库的文档。
class DeleteRequest(BaseModel):
    url: str

#返回向量数据库中按某种标签统计的元数据数量。
class MetadataQueryResponse(BaseModel):
    metadata_counts: Dict[str, int]

'''
用于调试向量数据库（如 Qdrant、Chroma、Milvus）的数据点。
一个向量点（DebugPoint）包括：
id: 唯一标识符；
payload: 元数据；
vector: 实际的向量。
'''
class DebugPoint(BaseModel):
    id: str
    payload: Dict #payload: 元数据；
    vector: List[float]

class DebugResponse(BaseModel):
    points: List[DebugPoint]    



