from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from llm_service.mychain import llm_answer, get_history
from schemas import QuestionRequest, AnswerResponse, UrlInjectionRequest, MetadataQueryResponse, DeleteRequest
from vectordb import add_urls_to_vectorstore, get_metadata_counts, delete_by_metadata, get_chroma ,add_document_to_vectorstore
from config import settings
import os
import time
import logging
import sys

#设置成FALSE，防止PostHog 在没网的时候无限次请求
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY4"] = "False"


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() #设置日志
logging.basicConfig( #这行是配置全局日志系统的输出格式和最低等级
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),   # 输出到控制台
    ],
    force=True# ✅ 防止被 uvicorn 默认配置覆盖
)
logger = logging.getLogger("RAG_API")#创建一个名为 "agentic_rag_api" 的日志记录器；你后面可以用它记录项目内各种运行信息。


@asynccontextmanager
async def lifespan(app: FastAPI): #这段是 FastAPI 的应用生命周期管理函数（lifespan）
    # Startup logic
    logger.info("RAG 服务开始启动-------------")
    yield  # App runs here  FastAPI 在这里进入“正常运行状态”，处理请求。

    logger.info("关闭 RAG API.")

app = FastAPI(title="Agentic RAG API" , lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://localhost:80"],  #  # 前端页面的来源
    allow_origins=["*"],                  #  放开所有来源（调试时可用）
    allow_credentials=True,# 允许携带 Cookie / Authorization 头
    allow_methods=["*"],# 允许所有 HTTP 方法（GET/POST/PUT/DELETE...）
    allow_headers=["*"],# 允许所有自定义 Header
)



@app.post("/uploadpdf")
async def upload_pdf(files: List[UploadFile] = File(...)):

    try:
        # files 是一个列表，里面每个都是 UploadFile 对象
        added_count, errors = await add_document_to_vectorstore(files)

        if errors:
            return {
                "message": f"添加 {added_count} 个块出现 {len(errors)} 个错误",
                "errors": errors,
                "status": "partial_success",
                "added_count": added_count,
            }

        return {
            "message": f"成功添加 {added_count} 个块",
            "status": "success",
            "added_count": added_count,
        }

        return {"message": f"成功上传 {len(files)} 个文件"}

    except Exception as e:
        logger.exception("失败在接口 /uploadpdf: %s", e)
        raise HTTPException(status_code=500, detail="文件服务错误")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    向 Agentic RAG 管道提问。
    返回漂亮打印的执行跟踪作为透明度的答案。
    """
    start = time.perf_counter()
    try:
        answer_text = llm_answer(request.question)

        elapsed = time.perf_counter() - start#计算耗时
        return AnswerResponse(answer=answer_text, processing_time=f"{elapsed:.2f}")
    except Exception as e:
        logger.exception("失败在接口 /ask: %s", e)
        raise HTTPException(status_code=500, detail="服务错误")


@app.get("/history")
async def history():
    return get_history()

@app.post("/inject") #向数据库插入数据
async def inject_urls(request: UrlInjectionRequest, background_tasks: BackgroundTasks):
    """
    将URL注入向量存储，然后在后台安排检索器刷新。如果部分URL失败，返回部分成功信息。
    """
    try:
        if not request.urls:
            raise HTTPException(status_code=400, detail="没有提交 URL")

        added_count, errors = add_urls_to_vectorstore(request.urls)


        if errors:
            return {
                "message": f"添加 {added_count} 个块出现 {len(errors)} 个错误",
                "errors": errors,
                "status": "partial_success",
                "added_count": added_count,
            }

        return {
            "message": f"成功添加 {added_count} 个块",
            "status": "success",
            "added_count": added_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("意外错误在 /inject: %s", e)
        raise HTTPException(status_code=500, detail="网络意外错误")


@app.post("/delete_by_metadata")
async def delete_by_metadata_endpoint(request: DeleteRequest, background_tasks: BackgroundTasks):
    """
    按元数据值（例如 URL）删除向量。
    始终在后台安排检索器刷新。
    """
    try:
        if not request.url:
            raise HTTPException(status_code=400, detail="确实 'url' 请求")

        deleted_count = delete_by_metadata(request.url)


        if deleted_count > 0:
            return {
                "message": f"Successfully deleted {deleted_count} chunks with metadata '{request.url}'",
                "deleted_count": deleted_count,
                "status": "success",
            }
        else:
            return {
                "message": f"No chunks found with metadata '{request.url}'",
                "deleted_count": 0,
                "status": "no_match",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("错误在接口 /delete_by_metadata: %s", e)
        raise HTTPException(status_code=500, detail="网络意外错误")



@app.get("/metadata/counts", response_model=MetadataQueryResponse)
async def get_metadata_counts_endpoint():
    """按元数据获取块计数"""
    try:
        counts = get_metadata_counts()
        return MetadataQueryResponse(metadata_counts=counts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/debug/points")
async def debug_points(limit: int = 50):
    """调试终结点以查看数据库中存储的内容"""
    try:
        vectorstore = get_chroma()
        collection = vectorstore._collection

        results = collection.get(
            limit=limit,
            include=["metadatas","documents"]  # 包含元数据和文档内容
        )


        debug_text = ""
        for i, doc_id in enumerate(results['ids']):
            source = results['metadatas'][i].get('source', '') if i < len(results['metadatas']) else ''
            content_preview = results['documents'][i][:200] + "..." if i < len(results['documents']) else ""

            debug_text += f"数据源：{source}\n内容：{content_preview}\n\n"

        return debug_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.get("/api/config")
async def get_config():
    """
    加载配置文件在前端显示
    """
    ss = settings.EMBEDDINGS_MODEL
    return {
        "llm_model": settings.LLM_MODEL,
        "embeddings_model": ss.rsplit("\\", 1)[-1]#将字符串分割，取右面
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)