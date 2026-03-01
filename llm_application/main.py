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