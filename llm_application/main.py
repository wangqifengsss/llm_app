# 第1、2、3天的代码（保留，无需修改，新增千问模型相关配置）
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from openai import OpenAI  # 千问模型兼容openai库，无需额外导入

app = FastAPI(
    title="大模型应用开发接口",
    version="1.0"
)

# 新增：配置魔塔社区千问模型（替换原OpenAI配置，重点！）
client = OpenAI(
    api_key="ms-bb133002-a51a-4276-9e1b-c95c7564a45a",  # 替换为你自己的ModelScope Access Token
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
        # 新增：处理边界情况1：用户提问为空
        if not request.question.strip():
            return {
                "status": "error",
                "code": 400,
                "user_id": request.user_id,
                "question": request.question,
                "history": request.history,
                "model": request.model,
                "answer": "请输入有效的提问，不能为空！"
            }

        # 新增：处理边界情况2：历史对话格式错误（确保每个元素有question和answer字段）
        valid_history = []
        for item in request.history:
            if "question" in item and "answer" in item and item["question"].strip() and item["answer"].strip():
                valid_history.append(item)
            else:
                # 过滤格式错误的历史对话，不影响接口运行
                continue

        # 拼接请求参数（使用过滤后的有效历史对话，适配千问模型）
        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        for item in valid_history:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["answer"]})
        messages.append({"role": "user", "content": request.question})

        # 调用魔塔社区千问模型API（不变，可根据需求切换stream）
        response = client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B",
            messages=messages,
            stream=False
        )
        # 解析响应（适配流式/非流式）

        # llm_answer = ""
        # for chunk in response:
        #     if chunk.choices[0].delta.content:
        #         llm_answer += chunk.choices[0].delta.content
      # llm_answer = response.choices[0].message.content.strip()  # 非流式
        llm_answer = response.choices[0].message.content.strip()

        # 更新历史对话（使用有效历史对话）
        new_history = valid_history.copy()
        new_history.append({"question": request.question, "answer": llm_answer})

        return {
            "user_id": request.user_id,
            "question": request.question,
            "history": new_history,
            "model": request.model,
            "answer": llm_answer
        }

    except Exception as e:
        # 优化错误提示，更简洁易懂（适配千问模型常见报错）
        error_msg = str(e)
        if "invalid api key" in error_msg.lower():
            error_msg = "ModelScope Access Token无效，请检查并替换正确的Token"
        elif "Timeout" in error_msg:
            error_msg = "网络超时，请切换网络后重新尝试"
        elif "Model not found" in error_msg:
            error_msg = "模型ID错误，请确认填写为Qwen/Qwen3.5-35B-A3B"
        elif "rate limit" in error_msg.lower():
            error_msg = "调用频率过高，请间隔1-2秒后重新测试"
        return {
            "status": "error",
            "code": 500,
            "user_id": request.user_id,
            "question": request.question,
            "history": request.history,
            "model": request.model,
            "answer": f"接口调用失败：{error_msg}"
        }