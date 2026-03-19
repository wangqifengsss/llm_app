# utils/config.py
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()  # 自动读取项目根目录的.env文件

# 豆包API配置（替换为你自己的API密钥）
ZHIPU_CONFIG = {
    "api_key": "xxx",  # 你的智谱API密钥
    "api_base": "https://open.bigmodel.cn/api/paas/v4/",  # 智谱兼容API地址
    "model": "GLM-4-Flash-250414",  # 选用的智谱模型版本（支持工具调用，无需修改）
    "temperature": 0.7,     # 温度参数：0-1，越低越严谨，越高越创意（适配工具调用）
    "max_tokens": 2048      # 最大输出token（足够满足工具调用及回答生成）
}

# 通义千问API配置（替换为你自己的API密钥）
TONGYI_CONFIG = {
    "api_key": "xxx",    # 你的通义千问API密钥
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容API地址
    "model": "qwen3.5-flash",   # 选用的通义千问模型版本（支持工具调用，无需修改）
    "temperature": 0.7,
    "max_tokens": 2048
}

# 工具调用配置：定义支持的工具列表（后续Agent扩展核心，今日先实现2个基础工具）
TOOL_CALL_CONFIG = {
    "tools": [
        #工具1用来数学计算
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "用于解决数学计算问题，支持加减乘除、乘方、开方等运算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学计算表达式，例如：1+2*3、sqrt(16)、pow(2,3)"
                        }
                    },
                    "required": ["expression"]  # 必传参数（计算表达式）
                }
            }
        },
        #工具2用来获取当前时间
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "用于获取当前的系统时间，无需输入参数",
                "parameters": {
                    "type": "object",
                    "properties": {}  # 无参数
                }
            }
        },
        #工具3用来获取指定城市天气
        {
            "type": "function",
            "function": {
                "name": "search_current_weather",
                "description": "【必选】查询指定城市的实时天气。调用此工具时，必须从用户问题中提取城市名称作为location参数。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        #查询天气时需要提供位置，因此参数设置为location
                        "location": {
                            "type": "string",
                            "description": "【必须提供】要查询的城市名称，例如：北京、上海、广州"
                        }
                    },
                    "required": ["location"] # 必传参数（城市名称）
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "translate_cn_to_en",
                "description": "用于将中文文本翻译成英文",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "必传参数，需要翻译的中文文本，例如：'今天天气很好'、'我喜欢编程'",
                            "maxLength": 1000  # 限制文本长度，避免异常
                        }
                    },
                    "required": ["text"]  # 必传参数：text（中文文本）
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_memory",
                "description": "当用户查询记忆时优先调用这个工具，用于查询长期记忆中的工具调用历史",
                "parameters": {
                    "type": "object",
                    "properties": {}  # 无参数
                }
            }
        },

    ]
}
# 新增：记忆功能配置（贴合今日实操，可灵活调整）
MEMORY_CONFIG = {
    "short_term_memory_limit": 10,  # 短期记忆最大条数（与memory_manager.py一致）
    "long_term_memory_limit": 5,    # 大模型获取长期记忆的最大条数（避免信息冗余）
    "save_long_term_memory": True  # 是否开启长期记忆持久化（默认开启）
}