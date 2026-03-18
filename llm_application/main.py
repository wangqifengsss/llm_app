# 第1、2、3天的代码（保留，无需修改，新增千问模型相关配置）
from fastapi import FastAPI, Query,Body
from pydantic import BaseModel
import requests
from openai import OpenAI  # 千问模型兼容openai库，无需额外导入
from config import QWEN_TOKEN,MODEL_ID,SERVICE_PORT,MAX_HISTORY_ROUNDS
from fastapi.middleware.cors import CORSMiddleware


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
            model=MODEL_ID,
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

# 修改FastAPI实例，添加全局文档注释（衔接第4天代码）
app = FastAPI(
    title="LLM应用接口文档",
    description="大模型应用开发工程师带教学习（第6天），包含千问模型调用、多轮对话优化等接口，适配Qwen/Qwen3.5-35B-A3B模型",
    version="1.0.0"
)

# 新增：配置魔塔社区千问模型（替换原OpenAI配置，重点！）
client = OpenAI(
    api_key=QWEN_TOKEN,  # 替换为你自己的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn/v1/"  # 魔塔社区API固定地址，不可修改
)
#在创建app实例后，添加跨域配置（新增）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问（开发环境可用，生产环境可指定具体域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法
    allow_headers=["*"],  # 允许所有请求头
)

# 根接口（保留）
@app.get("/", description="根接口，用于测试服务是否正常启动", summary="服务健康检测接口")
def root():
    return {"message": "Hello World"}


# 路径参数接口（保留）
@app.get("/user/{user_id}", description="根据用户ID查询用户信息（测试接口），返回用户ID和用户名", summary="用户查询测试接口")
def get_user(user_id: int):
    return {
        "status": "success",
        "user_id": user_id,
        "username": f"user_{user_id}",
        "message": f"成功查询到ID为{user_id}的用户"
    }


# 第3天定义的请求体模型（保留，无需修改，直接复用）
class LLMQueryRequest(BaseModel):
    user_id: int = Query(..., description="用户ID，整数类型，必填")
    question: str = Query(..., description="用户当前提问，字符串类型，必填，不可为空")
    history: list = Query([], description="历史对话列表，默认空列表，每个元素需包含question（用户提问）和answer（助手回答）字段")
    model: str = Query("Qwen/Qwen3.5-35B-A3B", description="模型ID，默认使用千问模型Qwen/Qwen3.5-35B-A3B")


# 第3天定义的响应模型（保留，无需修改，规范返回格式）
class LLMQueryResponse(BaseModel):
    user_id: int = Query(..., description="用户ID，与请求参数一致")
    question: str = Query(..., description="用户当前提问，与请求参数一致")
    history: list = Query(..., description="完整的历史对话列表（包含当前提问和回答）")
    model: str = Query(..., description="使用的模型ID")
    answer: str = Query(..., description="千问模型的回答，若报错则返回错误提示")
    status: str = Query(None, description="接口状态，success=成功，error=失败，可选")
    code: int = Query(None, description="错误码，400=客户端错误，500=服务端错误，可选")


# 修改：对接魔塔社区千问模型API的接口（替换第3天的/llm/query接口，适配你的代码）
# 修改：/llm/query接口，添加规范注释
@app.post("/llm/query",
          response_model=LLMQueryResponse,
          description="千问模型调用接口，支持多轮对话，已优化长对话报错问题，返回模型回答及完整历史对话",
          summary="千问模型多轮对话接口")
def llm_query(request: LLMQueryRequest=Body(...,description="请求参数，包含用户ID、提问、历史对话、模型ID，历史对话需包含question和answer字段")):
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

        # 截断历史对话：保留最近MAX_HISTORY_ROUNDS轮有效对话
        truncated_history = valid_history[-MAX_HISTORY_ROUNDS:] if len(valid_history) > MAX_HISTORY_ROUNDS else valid_history

        # 调用封装函数
        model_response = call_qwen_model(
            question=request.question,
            history=truncated_history,# 替换为截断后的历史对话
            token=QWEN_TOKEN
        )

        # 处理模型调用结果
        if model_response["status"] == "success":
            # 调用成功，更新完整的历史对话（不截断，前端显示全部）
            new_history = valid_history.copy()  # 仍使用valid_history，保留完整对话
            new_history.append({"question": request.question, "answer": model_response["answer"]})
            return {
                "user_id": request.user_id,
                "question": request.question,
                "history": new_history,  # 返回完整历史对话，便于前端展示
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



@app.get("/llm/test", description="测试千问模型封装函数的可复用性，返回模型直接回答", summary="千问模型复用测试接口")
def test_qwen():
    qwen_token = QWEN_TOKEN
    response = call_qwen_model(
        question="什么是Python？",
        history=[],
        token=qwen_token
    )
    return response