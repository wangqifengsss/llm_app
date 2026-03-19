# api/tongyi_openai_api.py
import os
from openai import OpenAI
from utils.config import TONGYI_CONFIG,TOOL_CALL_CONFIG
from typing import Optional, List, Dict, Any
from fastapi import HTTPException


# 初始化OpenAI兼容模式客户端（完全复用你提供的代码，适配config配置）
def init_tongyi_client() -> OpenAI:
    """初始化通义千问OpenAI兼容版客户端"""
    try:
        client = OpenAI(
            # 优先读取环境变量，未配置则使用config中的密钥
            api_key=os.getenv("DASHSCOPE_API_KEY", TONGYI_CONFIG["api_key"]),
            base_url=TONGYI_CONFIG["api_base"],
        )
        return client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"客户端初始化失败：{str(e)[:50]}")


def call_tongyi_api(
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
) -> Dict[str, Any]:
    """
    通义千问API调用（OpenAI兼容版），与其他API封装统一接口风格
    支持普通问答、工具调用，适配Agent工具调用闭环
    :param messages: 对话消息列表，格式：[{"role": "user/assistant/tool", "content": "内容"}]
    :param tools: 工具列表（工具调用时必填，格式同TOOL_CALL_CONFIG中的tools）
    :param tool_choice: 工具选择方式，可选"auto"（自动选择）、"none"（不调用）、具体工具名
    :return: 接口响应JSON数据（与OpenAI响应格式一致）
    """
    client = init_tongyi_client()
    try:
        # 构建请求参数，兼容OpenAI接口规范，同时适配工具调用
        params = {
            "model": TONGYI_CONFIG["model"],
            "messages": messages,
            "temperature": TONGYI_CONFIG["temperature"],
            "max_tokens": TONGYI_CONFIG["max_tokens"]
        }
        # 新增工具调用相关参数（适配Agent工具调用逻辑）
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # 调用OpenAI兼容接口（复用你提供的chat.completions.create方法）
        completion = client.chat.completions.create(**params)
        # 转换为JSON格式，与其他API封装返回格式保持一致，便于统一解析
        return completion.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"通义千问（OpenAI兼容版）API调用失败：{str(e)[:50]}")


# 测试函数：验证代码可用性（与你提供的测试逻辑一致，可直接运行）
if __name__ == "__main__":
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ]
        # 测试普通问答（无工具调用）
        result = call_tongyi_api(messages=messages, tool_choice="none")
        print("=== 通义千问（OpenAI兼容版）测试结果 ===")
        print(result)
        # 测试工具调用（时间查询）
        tool_messages = [{"role": "user", "content": "华盛顿的天气"}]
        tool_result = call_tongyi_api(
            messages=tool_messages,
            tools=TOOL_CALL_CONFIG["tools"],
            tool_choice="auto"
        )
        print("\n=== 工具调用测试结果 ===")
        print(tool_result)
    except Exception as e:
        print(f"测试失败：{str(e)}")