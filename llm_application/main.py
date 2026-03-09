# 第1、2、3天的代码（保留，无需修改，新增千问模型相关配置）
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from openai import OpenAI  # 千问模型兼容openai库，无需额外导入
from config import QWEN_TOKEN


# 新增：千问模型API调用封装函数（单一职责，只负责调用模型、返回回答）
def call_qwen_model(question: str, history: list, token: str) -> dict:
    try:
        # 原有拼接messages、调用模型、解析响应的逻辑不变...
        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        for item in history:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["answer"]})
        messages.append({"role": "user", "content": question})

        client = OpenAI(
            api_key=token,
            base_url="https://api-inference.modelscope.cn/v1/"
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B",
            messages=messages,
            stream=False
        )

        llm_answer = response.choices[0].message.content.strip()
        return {"status": "success", "answer": llm_answer}

    except Exception as e:
        error_msg = str(e)
        # 新增千问模型专属报错，优化提示并区分错误类型
        if "invalid api key" in error_msg.lower():
            error_msg = "【Token错误】ModelScope Access Token无效或过期，请重新创建并替换"
        elif "Timeout" in error_msg:
            error_msg = "【网络错误】调用超时，请切换手机热点、关闭防火墙后重新尝试"
        elif "Model not found" in error_msg:
            error_msg = "【模型错误】模型ID填写错误，请严格填写Qwen/Qwen3.5-35B-A3B"
        elif "rate limit" in error_msg.lower():
            error_msg = "【频率错误】调用频率过高，请间隔1-2秒后重新测试，避免频繁请求"
        elif "'ChatCompletion' object has no attribute 'stream'" in error_msg:
            error_msg = "【参数错误】千问模型不支持stream属性，请确保stream=False"
        elif "messages must be an array of objects" in error_msg:
            error_msg = "【参数错误】历史对话格式错误，需包含question和answer字段"
        elif "Insufficient quota" in error_msg or "quota" in error_msg.lower():
            error_msg = "【额度错误】千问模型调用额度不足，请降低调用频率或等待额度重置"
        else:
            error_msg = f"【未知错误】模型调用失败：{error_msg[:100]}"  # 截取前100字符，避免冗余
        return {"status": "error", "answer": error_msg}

app = FastAPI(
    title="大模型应用开发接口",
    version="1.0"
)

# 新增：配置魔塔社区千问模型（替换原OpenAI配置，重点！）
client = OpenAI(
    api_key=QWEN_TOKEN,  # 替换为你自己的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn/v1/"  # 魔塔社区API固定地址，不可修改
)


# 根接口（保留）
@app.get("/")
def read_root():
    return {"message": "Hello World! 欢迎学习大模型应用开发", "status": "success"}


# 路径参数接口（保留）
@app.get("/user/{user_id}")
def get_user(user_id: int):
    return {
        "status": "success",
        "user_id": user_id,
        "username": f"user_{user_id}",
        "message": f"成功查询到ID为{user_id}的用户"
    }


# 第3天定义的请求体模型（保留，无需修改，直接复用）
class LLMQueryRequest(BaseModel):
    question: str  # 必选字段，用户当前提问（核心参数）
    user_id: int  # 必选字段，用户ID
    history: list = []  # 可选字段，历史对话列表，默认空列表
    model: str = "gpt-3.5-turbo"  # 可选字段，默认模型，不影响千问调用（可保留）


# 第3天定义的响应模型（保留，无需修改，规范返回格式）
class LLMQueryResponse(BaseModel):
    status: str = "success"
    code: int = 200
    user_id: int
    question: str
    history: list
    model: str
    answer: str


# 修改：对接魔塔社区千问模型API的接口（替换第3天的/llm/query接口，适配你的代码）
@app.post("/llm/query", response_model=LLMQueryResponse)
def llm_query(request: LLMQueryRequest):
    try:
        # 边界情况1：空提问（客户端错误，code=400）
        if not request.question.strip():
            return {
                "status": "error",
                "code": 400,
                "user_id": request.user_id,
                "question": request.question,
                "history": request.history,
                "model": request.model,
                "answer": "【客户端错误】请输入有效的提问，不能为空！"
            }

        # 边界情况2：历史对话格式全部错误（客户端错误，code=400）
        valid_history = []
        for item in request.history:
            if "question" in item and "answer" in item and item["question"].strip() and item["answer"].strip():
                valid_history.append(item)
        if len(request.history) > 0 and len(valid_history) == 0:
            return {
                "status": "error",
                "code": 400,
                "user_id": request.user_id,
                "question": request.question,
                "history": request.history,
                "model": request.model,
                "answer": "【客户端错误】所有历史对话格式错误，需包含question和answer字段"
            }

        # 调用封装函数
        model_response = call_qwen_model(
            question=request.question,
            history=valid_history,
            token=QWEN_TOKEN
        )

        # 处理模型调用结果
        if model_response["status"] == "success":
            new_history = valid_history.copy()
            new_history.append({"question": request.question, "answer": model_response["answer"]})
            return {
                "user_id": request.user_id,
                "question": request.question,
                "history": new_history,
                "model": request.model,
                "answer": model_response["answer"]
            }
        else:
            # 判断错误类型，返回对应错误码
            if "【客户端错误】" in model_response["answer"]:
                code = 400
            else:
                code = 500
            return {
                "status": "error",
                "code": code,
                "user_id": request.user_id,
                "question": request.question,
                "history": valid_history,
                "model": request.model,
                "answer": model_response["answer"]
            }

    # 捕获接口自身的异常（如参数类型错误）
    except Exception as e:
        return {
            "status": "error",
            "code": 400,
            "user_id": request.user_id if hasattr(request, "user_id") else 0,
            "question": request.question if hasattr(request, "question") else "",
            "history": request.history if hasattr(request, "history") else [],
            "model": request.model if hasattr(request, "model") else "",
            "answer": f"【客户端错误】接口请求异常：{str(e)[:50]}"
        }



@app.get("/llm/test")
def test_qwen():
    qwen_token = QWEN_TOKEN
    response = call_qwen_model(
        question="什么是Python？",
        history=[],
        token=qwen_token
    )
    return response