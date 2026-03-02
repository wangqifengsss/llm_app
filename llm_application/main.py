# 1. 导入FastAPI模块（核心步骤，固定写法）
from fastapi import FastAPI

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
# 新增：带查询参数的接口（模拟大模型问答，接收用户提问）
# 路由：/query，无路径参数，查询参数通过 ?question=xxx 拼接
@app.get("/query")
def query_llm(question: str, user_id: int = 1001):  # user_id是可选查询参数，默认值1001
    # 模拟大模型回答，后续会替换成真实的大模型API调用，今日先返回固定格式
    return {
        "status": "success",
        "user_id": user_id,  # 可选查询参数，可传递也可不用传递
        "question": question,  # 接收用户的提问（查询参数）
        "answer": f"模拟大模型回答：{question} 的答案是..."  # 模拟回答
    }
