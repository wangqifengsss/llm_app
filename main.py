# main_demo.py
from api.zhipu_api import call_zhipu_api
from api.tongyi_api import call_tongyi_api
from utils.tool_manager import tool_manager  # 工具管理器（沿用第13天）
from utils.memory_manager import memory_manager  # 新增：导入记忆管理器
from utils.config import MEMORY_CONFIG ,TOOL_CALL_CONFIG # 新增：导入记忆配置
import json
import math
from fastapi import FastAPI, HTTPException

# 初始化FastAPI应用（沿用第13天，无需修改）
app = FastAPI(title="Agentic RAG 多工具调度+记忆功能接口", version="1.1")


def execute_tool(tool_name: str, parameters: dict) -> str:
    """
    执行工具（沿用第13天，补充记忆记录，无需修改原有逻辑）
    :param tool_name: 工具名
    :param parameters: 工具调用参数
    :return: 工具执行结果（字符串）
    """
    # 1. 校验工具是否存在
    tool = tool_manager.get_tool(tool_name)
    if not tool:
        result = f"工具执行失败：不支持的工具【{tool_name}】"
        tool_manager.log_tool_call(tool_name, parameters, result)
        return result

    # 2. 校验工具参数
    validate_result = tool_manager.validate_tool_parameters(tool_name, parameters)
    if validate_result:
        result = validate_result
        tool_manager.log_tool_call(tool_name, parameters, result)
        return result

    # 3. 执行对应工具（沿用第13天逻辑，包含翻译工具）
    try:
        if tool_name == "calculator":
            expression = parameters.get("expression")
            result = eval(expression, {"__builtins__": {}}, math.__dict__)
            result_str = f"计算结果：{expression} = {result}"
        elif tool_name == "get_current_time":
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_str = f"当前系统时间：{current_time}"
        elif tool_name == "search_current_weather":
            location = parameters.get("location")
            import requests
            weather_url = f"https://wttr.in/{location}?format=3"
            response = requests.get(weather_url, timeout=10)
            response.raise_for_status()
            result_str = f"【{location}实时天气】{response.text.strip()}"
        elif tool_name == "translate_cn_to_en":
            text = parameters.get("text")
            import requests
            translate_url = f"https://api.mymemory.translated.net/get?q={text}&langpair=zh|en"
            response = requests.get(translate_url, timeout=10)
            response.raise_for_status()
            translate_data = response.json()
            if translate_data.get("responseData") and translate_data["responseData"].get("translatedText"):
                en_text = translate_data["responseData"]["translatedText"]
                result_str = f"翻译结果（中文→英文）：\n中文原文：{text}\n英文译文：{en_text}"
            else:
                result_str = f"翻译失败：未获取到有效翻译结果，请检查输入文本是否为中文"
        elif tool_name == "query_memory":
            long_term_memory = memory_manager.get_long_term_memory(limit=MEMORY_CONFIG["long_term_memory_limit"])
            if not long_term_memory:
                result_str = "长期记忆为空，暂无工具调用历史"
            else:
                result_str = "长期记忆中的最近5条工具调用历史：\n"
                for i, memory in enumerate(long_term_memory):
                    result_str += f"{i+1}. {memory['content'][:30]}...\n"
                result_str=result_str.strip() # 去除最后的换行符
        else:
            result_str = f"工具执行失败：未实现工具【{tool_name}】的执行逻辑"
    except Exception as e:
        result_str = f"工具执行失败：{str(e)[:50]}"

    # 4. 记录工具调用日志（沿用第13天）
    tool_manager.log_tool_call(tool_name, parameters, result_str)

    # 新增：将工具调用记录添加到长期记忆（跨对话保存）
    if MEMORY_CONFIG["save_long_term_memory"]:
        # 优化记忆内容，明确记录工具调用的核心信息（重点突出天气查询的城市，便于大模型快速提取）
        if tool_name == "search_current_weather":
            # 单独优化天气工具的记忆格式，明确标注查询的城市
            memory_content = f"用户调用search_current_weather工具，查询了【{parameters.get('location')}】的天气，参数：{json.dumps(parameters, ensure_ascii=False)}，结果：{result_str[:30]}..."
        else:
            memory_content = f"调用工具【{tool_name}】，参数：{json.dumps(parameters, ensure_ascii=False)}，结果：{result_str[:30]}..."
        memory_manager.add_long_term_memory({
            "type": "tool_call",
            "content": memory_content
        })

    return result_str


def agent_run(question: str, model_type: str = "zhipu") -> str:
    """
    优化版Agent核心逻辑：整合记忆功能（短期+长期）+ 多工具调度，实现全闭环
    核心升级：结合记忆，让Agent能记住上下文和历史工具调用，无需重复输入关键信息
    """
    # 1. 记忆初始化：清空当前对话的短期记忆（切换对话时）
    memory_manager.clear_short_term_memory()

    # 2. 感知：接收用户提问，添加到短期记忆
    memory_manager.add_short_term_memory({"role": "user", "content": question})

    # 3. 规划：获取所有已注册工具 + 记忆信息，让大模型结合记忆判断工具调用
    all_tools = tool_manager.get_all_tools()
    # 获取长期记忆（最近N条，避免信息过多）
    long_term_memory = memory_manager.get_long_term_memory(limit=MEMORY_CONFIG["long_term_memory_limit"])
    # 获取短期记忆（当前对话上下文）
    short_term_memory = memory_manager.get_short_term_memory()

    # 构建大模型提示词：结合记忆信息，引导大模型使用记忆、判断工具调用
    system_prompt = f"""
你是一个具备记忆功能的Agent，需要结合短期记忆（当前对话）和长期记忆（历史记录），调用工具来完成用户提问。
## 一、核心能力
你可以调用以下工具获取实时信息：
{json.dumps(TOOL_CALL_CONFIG['tools'], ensure_ascii=False, indent=2)}

## 二、当前上下文
### 短期记忆（当前对话）
{json.dumps(short_term_memory, ensure_ascii=False, indent=2)}

### 长期记忆（历史记录）
{json.dumps(long_term_memory, ensure_ascii=False, indent=2)}

## 三、强制工具调用规则

### ⏰ 时间查询
当用户提到：时间、几点、钟、日期
必须调用：get_current_time()
禁止回答："我无法提供时间"

### ☀️ 天气查询
当用户提到：天气、气温、温度、下雨、晴天
必须调用：search_current_weather(location="城市名")
禁止回答："请查看天气预报"

### 🌐 翻译查询
当用户提到：翻译、英文、怎么说
必须调用：translate_cn_to_en(text="中文文本")
禁止回答："我无法翻译"

### 🧮 计算查询
当用户提到：计算、等于、多少、加减乘除
必须调用：calculator(expression="数学表达式")
禁止回答："请自己计算"

### 📚 工具调用查询
当用户提到：工具、调用、记录、历史
必须调用：query_memory()
禁止回答："我无法提供工具调用记录"

## 四、执行流程
1. 识别用户问题类型
2. 提取必要参数
3. 调用对应工具
4. 根据工具结果生成回答

记住：你的使命是调用工具，不是拒绝回答！
"""

    # 构建对话消息：系统提示词（包含记忆）+ 短期记忆（用户提问）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # 调用对应模型API（沿用第13天逻辑，仅优化提示词）
    try:
        if model_type == "zhipu":
            response = call_zhipu_api(
                messages=messages,
                tools=all_tools,
                tool_choice="auto"  # 让大模型自动选择工具（支持多工具）
            )
        elif model_type == "tongyi":
            response = call_tongyi_api(
                messages=messages,
                tools=all_tools,
                tool_choice="auto"
            )
        else:
            return "错误：不支持的模型类型，仅支持zhipu、tongyi"
    except HTTPException as e:
        return f"API调用失败：{e.detail}"
    except Exception as e:
        return f"API调用失败：{str(e)[:50]}"

    # 4. 解析响应：处理大模型返回的工具调用（支持多个工具调用，沿用第13天逻辑）
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls", [])

    # 5. 行动：执行工具（支持多个工具，循环执行，新增短期记忆记录）
    if not tool_calls:
        # 无需调用工具，直接返回大模型回答，同时将回答添加到短期记忆
        answer = message.get("content", "无回答内容")
        memory_manager.add_short_term_memory({"role": "assistant", "content": answer})
        return answer

    # 循环执行所有调用的工具（核心：多工具调度+短期记忆记录）
    for tool_call in tool_calls:
        tool_name = tool_call["function"].get("name")
        tool_parameters = json.loads(tool_call["function"].get("arguments", "{}"))
        print(f"\n=== 大模型自主选择工具：{tool_name}，参数：{tool_parameters} ===")

        # 执行工具，获取结果
        tool_result = execute_tool(tool_name, tool_parameters)
        print(f"=== 工具执行结果：{tool_result} ===")

        # 将工具调用和结果添加到短期记忆（供大模型后续决策使用）
        memory_manager.add_short_term_memory({
            "role": "tool",
            "content": f"工具【{tool_name}】调用结果：{tool_result}"
        })

        # 将工具调用结果添加到对话上下文
        messages.append(message)
        messages.append({
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_call["id"]
        })

    # 6. 反馈：调用大模型，基于记忆+工具结果，生成最终回答
    try:
        if model_type == "zhipu":
            final_response = call_zhipu_api(messages=messages)
        elif model_type == "tongyi":
            final_response = call_tongyi_api(messages=messages)
        else:
            return "错误：不支持的模型类型，无法生成最终回答"
    except Exception as e:
        return f"最终回答生成失败：{str(e)[:50]}"

    final_answer = final_response.get("choices", [{}])[0].get("message", {}).get("content", "生成最终回答失败")
    # 新增：将最终回答添加到短期记忆
    memory_manager.add_short_term_memory({"role": "assistant", "content": final_answer})
    return final_answer


# FastAPI接口（优化：新增记忆功能支持，返回记忆信息，沿用原有接口地址）
@app.get("/agent/query")
def agent_query(question: str, model_type: str = "zhipu"):
    try:
        result = agent_run(question=question, model_type=model_type)
        return {
            "code": 200,
            "message": "success",
            "data": {
                "answer": result,
                "tool_logs": tool_manager.tool_call_logs,
                "short_term_memory": memory_manager.get_short_term_memory(),  # 新增：返回短期记忆
                "long_term_memory": memory_manager.get_long_term_memory()  # 新增：返回长期记忆
            }
        }
    except Exception as e:
        return {"code": 500, "message": f"接口调用失败：{str(e)}", "data": None}


# 主程序：测试记忆功能+多工具调度闭环（可直接运行，重点测试连续对话场景）
if __name__ == "__main__":
    print("=== 大模型Agent多工具调度+记忆功能Demo（第14天核心）===")
    print("支持工具：calculator、search_current_weather、get_current_time、translate_cn_to_en")
    print("支持模型：zhipu、tongyi（默认zhipu）")
    print("核心功能：短期记忆（当前对话）、长期记忆（跨对话保存）")
    print("测试场景1（短期记忆）：先查北京天气，再问‘刚才查的城市天气怎么样’")
    print("测试场景2（长期记忆）：关闭程序重新运行，输入‘我之前查过哪个城市的天气’")
    print("输入 'exit' 即可退出程序\n")

    while True:
        user_question = input("请输入你的问题：")
        if user_question.lower() == "exit":
            print("程序退出！")
            break

        model_choice = input("选择模型（输入 zhipu/tongyi，默认zhipu）：") or "zhipu"
        final_answer = agent_run(question=user_question, model_type=model_choice)
        print(f"\n=== 最终回答：{final_answer} ===\n")
        # 打印当前短期记忆（便于验证）
        print("=== 当前短期记忆 ===")
        for mem in memory_manager.get_short_term_memory():
            print(f"{mem['time']} | {mem['role']}: {mem['content'][:50]}...")
        # 新增：打印当前长期记忆（便于验证长期记忆是否正常存储）
        print("=== 当前长期记忆（最近5条） ===")
        for mem in memory_manager.get_long_term_memory():
            print(f"{mem['time']} | {mem['type']}: {mem['content'][:50]}...")
        print("=" * 50 + "\n")