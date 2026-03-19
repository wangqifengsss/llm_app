# main.py
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
        elif tool_name == "get_city_code":
            # 功能：根据城市名称获取城市编码，调用公开免费API
            city = parameters.get("city")
            if not city:
                return "❌ 城市编码查询失败：未传入城市名称（必传参数）"
            # 调用公开API（无需注册，直接可用）
            import requests
            try:
                url = f"https://restapi.amap.com/v3/config/district?keywords={city}&subdistrict=0&key=66ec065eaff41cb092090185b798f9bc"
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # 抛出HTTP请求异常
                data = response.json()
                # 解析API返回结果，提取城市编码
                if data.get("status") == "1" and "districts" in data:
                    city_info = data["districts"][0]  # 取第一个匹配的城市
                    city_code = city_info.get("adcode")
                    city_name = city_info.get("name")
                    return f"【城市编码查询结果】城市：{city_name}，行政区划编码：{city_code}（来源：公开天气API）"
                else:
                    return f"❌ 城市编码查询失败：未查询到{city}的相关信息，请检查城市名称是否正确"
            except Exception as e:
                return f"❌ 城市编码查询失败：{str(e)[:30]}（API调用异常）"
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


def split_complex_task(user_question: str, model_type: str = "zhipu") -> list:
    """
    辅助Agent拆解复杂任务，将用户复杂提问拆分为多个可执行的子任务
    :param user_question: 用户原始复杂提问
    :param model_type: 模型类型（zhipu/tongyi）
    :return: 拆解后的子任务列表（每个子任务是字典，包含task、tool、parameters）
    """
    # 构建任务拆解提示词，引导大模型按指定格式拆解任务
    split_prompt = """
你是一个任务拆解助手，需要将用户的复杂提问，拆解为多个可执行的子任务，每个子任务对应一个可用工具，要求如下：
1.  可用工具：calculator（计算）、search_current_weather（查询天气）、translate_cn_to_en（中译英）、query_memory（查询记忆）get_city_code（查询城市编码）；
2.  拆解规则：
    - 每个子任务只能对应一个工具，不能一个子任务调用多个工具；
    - 子任务按执行顺序排列（先执行的在前，后执行的在后，确保逻辑连贯）；
    - 子任务需明确“任务描述（task）、所需工具（tool）、工具参数（parameters）”，参数缺失时标注“需后续获取”；
    - 若需要复用记忆中的信息（如之前查询过的城市），子任务中需标注“复用长期记忆中的XX信息”；
    - 无需多余解释，仅返回拆解后的子任务列表，格式为JSON数组，示例：
      [
          {"task": "查询北京天气", "tool": "search_current_weather", "parameters": {"location": "北京"}},
          {"task": "将北京天气结果翻译成英文", "tool": "translate_cn_to_en", "parameters": {"text": "北京实时天气：温度10°C，天气晴"}},
          {"task": "计算北京温度的2倍", "tool": "calculator", "parameters": {"expression": "10*2"}}
      ]
3.  用户当前复杂提问：""" + user_question

    # 调用模型，获取拆解后的子任务列表
    messages = [{"role": "user", "content": split_prompt}]
    try:
        if model_type == "zhipu":
            response = call_zhipu_api(messages=messages, tools=[], tool_choice="none")  # 不调用工具，仅拆解任务
        elif model_type == "tongyi":
            response = call_tongyi_api(messages=messages, tools=[], tool_choice="none")
        else:
            return [{"task": "任务拆解失败", "tool": "none", "parameters": {}}]
    except Exception as e:
        return [{"task": f"任务拆解失败：{str(e)[:30]}", "tool": "none", "parameters": {}}]

    # 解析模型返回的子任务列表（确保格式正确）
    try:
        split_tasks = json.loads(response["choices"][0]["message"]["content"])
        # 校验子任务格式，确保每个子任务包含task、tool、parameters
        for task in split_tasks:
            if not all(key in task for key in ["task", "tool", "parameters"]):
                return [{"task": "子任务格式错误", "tool": "none", "parameters": {}}]
        return split_tasks
    except Exception as e:
        return [{"task": f"子任务解析失败：{str(e)[:30]}", "tool": "none", "parameters": {}}]

    # 新增：JSON解析容错处理，解决Expecting value报错



def integrate_task_results(split_tasks: list, task_results: list) -> str:
    """
    整合子任务执行结果，生成最终回答
    :param split_tasks: 拆解后的子任务列表
    :param task_results: 每个子任务的执行结果列表（与子任务顺序一致）
    :return: 整合后的最终回答
    """
    if len(split_tasks) != len(task_results):
        return "任务执行异常：子任务数量与结果数量不匹配"

    final_answer = "复杂任务执行完成，结果如下：\n"
    for idx, (task, result) in enumerate(zip(split_tasks, task_results), 1):
        final_answer += f"\n{idx}. 子任务：{task['task']}\n   执行结果：{result}\n"

    return final_answer.strip()


def agent_run(question: str, model_type: str = "zhipu") -> str:
    """
    优化版Agent核心逻辑：整合记忆功能（短期+长期）+ 复杂任务拆解 + 多工具调度，实现全闭环
    核心升级：能自主拆解复杂任务，按顺序调用工具，整合结果，衔接第14天记忆功能
    """
    # 1. 记忆初始化：清空当前对话的短期记忆（切换对话时）
    memory_manager.clear_short_term_memory()

    # 2. 感知：接收用户提问，添加到短期记忆
    memory_manager.add_short_term_memory({"role": "user", "content": question})

    # 3. 规划：获取所有已注册工具 + 记忆信息，判断是否为复杂任务
    all_tools = tool_manager.get_all_tools()
    long_term_memory = memory_manager.get_long_term_memory(limit=MEMORY_CONFIG["long_term_memory_limit"])
    short_term_memory = memory_manager.get_short_term_memory()

    # 新增：判断是否为复杂任务（包含多个操作，需拆解）
    # 简单判断：提问中包含“和”“并”“然后”“再”等连接词，视为复杂任务
    complex_task_keywords = ["和", "并", "然后", "再", "同时", "依次"]
    is_complex_task = any(keyword in question for keyword in complex_task_keywords)

    # 4. 复杂任务拆解：若为复杂任务，先拆解为子任务
    if is_complex_task:
        print(f"\n=== 检测到复杂任务，开始拆解 ===")
        split_tasks = split_complex_task(question, model_type)
        # 记录任务拆解日志
        tool_manager.log_task_split(question, split_tasks)
        print(f"=== 任务拆解完成，共{len(split_tasks)}个子任务：{[task['task'] for task in split_tasks]} ===")

        # 校验拆解结果，若拆解失败，直接返回错误
        if split_tasks[0]["tool"] == "none":
            error_msg = split_tasks[0]["task"]
            memory_manager.add_short_term_memory({"role": "assistant", "content": error_msg})
            return error_msg

        # 执行所有子任务（按顺序执行，复用记忆和工具结果）
        task_results = []
        # 新增：子任务去重，避免重复执行（解决tongyi模型拆解重复问题）
        unique_tasks = []
        task_ids = set()
        for task in split_tasks:
            # 用任务名称作为唯一标识，去重
            task_key = task["task"] + task["tool"]
            if task_key not in task_ids:
                task_ids.add(task_key)
                unique_tasks.append(task)
        split_tasks = unique_tasks  # 替换为去重后的子任务列表
        for task in split_tasks:
            tool_name = task["tool"]
            tool_params = task["parameters"]

            # 优化：复用记忆中的信息（如之前查询过的城市、翻译结果）
            if "复用长期记忆" in task["task"]:
                # 从长期记忆中提取所需信息（以天气查询为例，可扩展）
                for mem in long_term_memory:
                    if mem["type"] == "tool_call" and "search_current_weather" in mem["content"]:
                        # 提取记忆中的城市名称
                        import re
                        location_match = re.search(r"【(.*?)】", mem["content"])
                        if location_match:
                            tool_params["location"] = location_match.group(1)
                            print(f"✅ 复用长期记忆，获取城市：{tool_params['location']}")

            # 新增：子任务失败重试逻辑（最多重试1次）
            retry_count = 0  # 重试次数计数器
            max_retry = 1  # 最大重试次数
            tool_result = ""
            while retry_count <= max_retry:
                try:
                    # 执行当前子任务的工具
                    print(f"\n=== 执行子任务：{task['task']}，调用工具：{tool_name}，参数：{tool_params} ====")
                    print(f"=== 第{retry_count + 1}次执行（重试次数：{retry_count}）===")
                    tool_result = execute_tool(tool_name, tool_params)
                    print(f"=== 子任务执行成功，结果：{tool_result[:30]}... ====")
                    break  # 执行成功，跳出重试循环
                except Exception as e:
                    retry_count += 1
                    error_msg = f"子任务执行失败：{str(e)[:30]}"
                    print(f"❌ {error_msg}，将进行第{retry_count}次重试（最多重试{max_retry}次）")
                    # 重试次数耗尽，记录失败信息
                    if retry_count > max_retry:
                        tool_result = error_msg + "（已重试1次，仍失败，跳过该子任务）"
                        print(f"❌ 重试次数耗尽，记录失败信息，继续执行下一个子任务")

            task_results.append(tool_result)
            # 将子任务执行结果添加到短期记忆，供后续子任务复用
            memory_manager.add_short_term_memory({
                "role": "tool",
                "content": f"子任务【{task['task']}】调用工具【{tool_name}】，结果：{tool_result}"
            })

        # 整合所有子任务结果，生成最终回答
        final_answer = integrate_task_results(split_tasks, task_results)
        memory_manager.add_short_term_memory({"role": "assistant", "content": final_answer})
        return final_answer

    # 5. 非复杂任务：沿用第14天逻辑，结合记忆处理提问（无需拆解）
    else:
        # 构建大模型提示词（沿用第14天优化后的提示词，新增任务拆解相关说明）
        system_prompt = f"""
你是一个具备记忆功能的Agent，需要结合短期记忆（当前对话）和长期记忆（历史记录），调用工具完成用户提问。
1.  短期记忆：{json.dumps(short_term_memory, ensure_ascii=False)}（当前对话的用户提问、工具调用结果）
2.  长期记忆：{json.dumps(long_term_memory, ensure_ascii=False)}（历史工具调用、用户偏好，重点关注tool_call类型的记忆）
3.  记忆使用规则：
    - 先判断用户提问是否与短期/长期记忆相关：若用户提问是全新的、与历史记忆（工具调用、用户偏好）无关（如首次查询某城市天气、首次翻译文本），则不依赖记忆，直接处理提问；
    - 若用户提问涉及“刚才”“之前”“之前查过”等上下文关联词汇，或需要复用历史信息（如重复查询、查询历史结果），则优先使用短期记忆，短期记忆无相关信息时，再使用长期记忆；
    - 避免无关记忆干扰：若用户提问与记忆无关，无需提及任何记忆相关内容，直接调用工具或直接回答。
4.  工具调用规则：
    - 调用工具前必须检查必传参数，参数缺失则询问用户补充；
    - 若用户提问可通过记忆直接回答（无需调用工具），则直接返回答案，不调用工具；
    - 若用户提问涉及“之前查询过的城市天气”“我之前查过哪个城市”等相关表述，必须优先读取长期记忆中的tool_call记录，重点筛选search_current_weather工具的调用记录，提取其中的location（城市名称），无需让用户重复输入；
    - 若长期记忆中存在search_current_weather工具的调用记录，直接提取城市名称，告知用户“你之前查询过XX城市的天气”，无需反问用户；
    - 若长期记忆中无相关tool_call记录（无天气查询历史），再询问用户“你之前查过哪个城市的天气呢？”；
    - 调用工具后，需结合工具结果和记忆，生成最终回答。
"""
        # 构建对话消息，调用模型处理非复杂任务
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        try:
            if model_type == "zhipu":
                response = call_zhipu_api(messages=messages, tools=all_tools, tool_choice="auto")
            elif model_type == "tongyi":
                response = call_tongyi_api(messages=messages, tools=all_tools, tool_choice="auto")
            else:
                return "错误：不支持的模型类型，仅支持zhipu、tongyi"
        except Exception as e:
            return f"API调用失败：{str(e)[:50]}"

        # 解析响应，执行工具（沿用第14天逻辑）
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            answer = message.get("content", "无回答内容")
            memory_manager.add_short_term_memory({"role": "assistant", "content": answer})
            return answer

        # 循环执行工具，记录记忆
        for tool_call in tool_calls:
            tool_name = tool_call["function"].get("name")
            tool_parameters = json.loads(tool_call["function"].get("arguments", "{}"))
            print(f"\n=== 大模型自主选择工具：{tool_name}，参数：{tool_parameters} ===")

            tool_result = execute_tool(tool_name, tool_parameters)
            print(f"=== 工具执行结果：{tool_result} ===")

            memory_manager.add_short_term_memory({
                "role": "tool",
                "content": f"工具【{tool_name}】调用结果：{tool_result}"
            })

            messages.append(message)
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call["id"]
            })

        # 生成最终回答
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
                "task_split_logs": tool_manager.task_split_logs,  # 新增：返回任务拆解日志
                "short_term_memory": memory_manager.get_short_term_memory(),
                "long_term_memory": memory_manager.get_long_term_memory()
            }
        }
    except Exception as e:
        return {"code": 500, "message": f"接口调用失败：{str(e)}", "data": None}


# 主程序：测试记忆功能+多工具调度闭环（可直接运行，重点测试连续对话场景）
# main_demo.py（主程序修改，可直接复制）
if __name__ == "__main__":
    print("=== 大模型Agent多工具调度+记忆功能+复杂任务拆解Demo（第15天核心）===")
    print("支持工具：calculator、search_current_weather、get_current_time、translate_cn_to_en、query_memory等")
    print("支持模型：zhipu、tongyi（默认zhipu）")
    print("核心功能：短期记忆、长期记忆、复杂任务拆解、多工具协同")
    print("测试场景1（复杂任务）：输入‘查询广州天气，翻译成英文，再计算温度的2倍’")
    print("测试场景2（记忆+拆解）：先查北京天气，再输入‘刚才查的城市天气，翻译成英文并计算温度+5’")
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
        print("=== 当前短期记忆 ====")
        for mem in memory_manager.get_short_term_memory():
            print(f"{mem['time']} | {mem['role']}: {mem['content'][:50]}...")
        # 打印当前长期记忆（便于验证）
        print("=== 当前长期记忆（最近5条） ====")
        for mem in memory_manager.get_long_term_memory():
            print(f"{mem['time']} | {mem['type']}: {mem['content'][:50]}...")
        # 新增：打印任务拆解日志（若有）
        if tool_manager.task_split_logs:
            print("=== 任务拆解日志（最近1条） ====")
            last_split_log = tool_manager.task_split_logs[-1]
            print(f"时间：{last_split_log['time']}")
            print(f"用户提问：{last_split_log['user_question']}")
            print(f"拆解子任务：{[task['task'] for task in last_split_log['split_tasks']]}")
        print("=" * 50 + "\n")