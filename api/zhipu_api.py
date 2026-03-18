# api/zhipu_api.py
import os
from openai import OpenAI
from utils.config import ZHIPU_CONFIG ,TOOL_CALL_CONFIG
from typing import Optional, List, Dict, Any
from fastapi import HTTPException  # 保持FastAPI依赖，适配后续接口开发


# 初始化智谱OpenAI兼容模式客户端（参考通义千问OpenAI兼容版逻辑，保持一致）
def init_zhipu_client() -> OpenAI:
    """初始化智谱大模型OpenAI兼容版客户端"""
    try:
        client = OpenAI(
            # 优先读取环境变量，未配置则使用config中的智谱密钥，适配企业级部署
            api_key=os.getenv("ZHIPU_API_KEY", ZHIPU_CONFIG["api_key"]),
            base_url=ZHIPU_CONFIG["api_base"],  # 智谱官方API地址本身支持OpenAI兼容格式
        )
        return client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"智谱客户端初始化失败：{str(e)[:50]}")


def call_zhipu_api(
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
) -> Dict[str, Any]:
    """
    智谱大模型API调用（OpenAI兼容版），与其他API封装统一接口风格
    支持普通问答、工具调用，适配Agent工具调用闭环，完全兼容OpenAI接口规范
    :param messages: 对话消息列表，格式：[{"role": "user/assistant/tool", "content": "内容"}]
    :param tools: 工具列表（工具调用时必填，格式同TOOL_CALL_CONFIG中的tools）
    :param tool_choice: 工具选择方式，可选"auto"（自动选择）、"none"（不调用）、具体工具名
    :return: 接口响应JSON数据（与OpenAI响应格式一致，便于统一解析）
    """
    client = init_zhipu_client()
    try:
        # 构建请求参数，兼容OpenAI接口规范，同时适配工具调用和智谱模型参数
        params = {
            "model": ZHIPU_CONFIG["model"],
            "messages": messages,
            "temperature": ZHIPU_CONFIG["temperature"],
            "max_tokens": ZHIPU_CONFIG["max_tokens"]
        }
        # 工具调用相关参数（与通义千问OpenAI兼容版一致，适配Agent工具调用逻辑）
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # 调用OpenAI兼容接口（与通义千问OpenAI兼容版调用方式一致，统一代码风格）
        completion = client.chat.completions.create(**params)
        # 转换为JSON格式，与其他API封装（智谱原版、通义千问原生/兼容版）返回格式保持统一
        return completion.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"智谱API调用失败：{str(e)[:50]}")


# 测试函数：验证代码可用性（参考通义千问测试逻辑，保持一致，可直接运行）
if __name__ == "__main__":
    try:
        # 测试1：普通问答（无工具调用）
        messages_qa = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "介绍一下智谱大模型"},
        ]
        result_qa = call_zhipu_api(messages=messages_qa, tool_choice="none")
        print("=== 智谱（OpenAI兼容版）普通问答结果 ===")
        print(result_qa.get("choices", [{}])[0].get("message", {}).get("content", "无结果"))

        # 测试2：工具调用（计算器，验证工具调用逻辑，与项目工具列表适配）
        messages_tool = [{"role": "user", "content": "北京现在的天气"}]
        result_tool = call_zhipu_api(
            messages=messages_tool,
            tools = TOOL_CALL_CONFIG["tools"],
            tool_choice="auto"  # 让大模型自动判断是否调用工具
        )
        print("\n=== 智谱（OpenAI兼容版）工具调用结果 ===")
        print(result_tool)
    except Exception as e:
        print(f"测试失败：{str(e)}")

# 补充说明（关键，衔接项目整体逻辑）：
# 1. 配置兼容：无需新增智谱配置，直接复用config.py中的ZHIPU_CONFIG，减少冗余
# 2. 接口统一：与通义千问OpenAI兼容版、原生版接口风格完全一致，后续切换模型无需修改业务代码
# 3. 功能兼容：支持普通问答、工具调用，完美适配main_demo.py中的Agent工具调用闭环
# 4. 部署兼容：支持环境变量配置密钥，贴合企业级部署规范，与FastAPI接口无缝集成