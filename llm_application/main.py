# 1. 导入FastAPI模块（核心步骤，固定写法）
from fastapi import FastAPI
from pydantic import BaseModel

# 2. 创建FastAPI应用实例（app是实例名，后续所有接口都基于这个实例）
app = FastAPI(
    title="大模型应用开发接口",  # 接口文档的标题（可选，让文档更规范）
    version="1.0"  # 接口版本（可选）
)

# 3. 定义接口：用@app.get("/")装饰器，定义接口路由和请求方法
# @app.get("/") 表示：接口地址是 http://127.0.0.1:8000/，请求方法是GET
@app.get("/")
def read_root():
    # 4. 接口函数：接口被访问时，会执行这个函数，返回函数的返回值
    return {"message": "Hello World! 欢迎学习大模型应用开发", "status": "success"}

# 新增：带路径参数的接口（查询用户信息）
# 路由：/user/{user_id}，user_id是路径参数，用于接收用户ID
@app.get("/user/{user_id}")
def get_user(user_id: int):  # : int 限制user_id必须是数字，更规范
    # 模拟根据用户ID查询信息，后续可对接数据库，今日先返回固定格式
    return {
        "status": "success",
        "user_id": user_id,  # 返回接收的路径参数
        "username": f"user_{user_id}",  # 模拟用户名
        "message": f"成功查询到ID为{user_id}的用户"
    }
# 新增1：定义请求体模型（用于接收复杂参数：提问+历史对话）
class LLMQueryRequest(BaseModel):
    question: str  # 必选字段，用户当前提问（核心参数）
    user_id: int   # 必选字段，用户ID
    history: list = []  # 可选字段，历史对话列表，默认空列表（无历史对话）
    # 可新增字段，比如对话类型、模型选择等，贴合后续大模型调用
    model: str = "gpt-3.5-turbo"  # 可选字段，默认模型，后续可对接真实模型

# 新增2：带请求体的接口（POST请求，用于大模型多轮对话，替换原/query接口）
@app.post("/llm/query")  # 用POST请求，传递复杂请求体
def llm_query(request: LLMQueryRequest):  # request接收请求体，类型是我们定义的模型
    # 模拟大模型多轮对话逻辑：结合历史对话和当前提问，返回回答
    # 可通过 request.字段名 获取请求体中的所有参数
    return {
        "status": "success",
        "user_id": request.user_id,
        "question": request.question,
        "history": request.history,
        "model": request.model,
        "answer": f"模拟大模型多轮回答：针对你的提问「{request.question}」，结合历史对话，答案是..."
    }


# 新增3：定义响应模型（规范接口返回格式，继承BaseModel）
class LLMQueryResponse(BaseModel):
    status: str = "success"  # 固定默认值，无需手动返回
    code: int =200   # 新增状态码，贴合企业接口规范（200=成功）
    user_id: int
    question: str
    history: list
    model: str
    answer: str
    # 可新增字段，比如响应时间，后续可扩展
    # create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 修改：带请求体+响应模型的接口（新增response_model参数）
@app.post("/llm/query", response_model=LLMQueryResponse)  # 指定响应模型
def llm_query(request: LLMQueryRequest):
    # 函数返回的内容，会自动按响应模型规范，多余字段会被过滤
    return {
        "user_id": request.user_id,
        "question": request.question,
        "history": request.history,
        "model": request.model,
        "answer": f"模拟大模型多轮回答：针对你的提问「{request.question}」，结合历史对话，答案是...",
        "extra": "这个多余字段会被响应模型过滤，不会返回" # 测试过滤效果
    # status和code无需手动返回，响应模型会自动填充默认值
    }
print("正在运行的 main.py 文件路径:", __file__)