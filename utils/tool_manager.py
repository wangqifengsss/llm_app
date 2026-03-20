"""
工具注册、管理核心文件
"""
# utils/tool_manager.py
from typing import Dict, List, Any, Optional
from utils.config import TOOL_CALL_CONFIG  # 沿用第12天工具配置
import json
from datetime import datetime


class ToolManager:
    """Agent工具管理类：实现工具注册、查询、参数校验，统一管理所有工具"""

    def __init__(self):
        # 初始化工具字典，key：工具名，value：工具详细配置（从config中读取）
        self.tools: Dict[str, Dict[str, Any]] = self._load_tools_from_config()
        # 工具调用日志，记录每次工具调用的详细信息（便于排查问题）
        self.tool_call_logs: List[Dict[str, Any]] = []
        self.task_split_logs = []  # 新增：任务拆解日志，记录Agent拆解任务的过程

    def _load_tools_from_config(self) -> Dict[str, Dict[str, Any]]:
        """从config.py中加载工具配置，转换为字典格式（便于快速查询）"""
        tools_dict = {}
        for tool in TOOL_CALL_CONFIG["tools"]:
            tool_name = tool["function"]["name"]
            tools_dict[tool_name] = tool  # 以工具名为key，存储完整工具配置
        return tools_dict

    def register_tool(self, tool: Dict[str, Any]) -> bool:
        """
        注册新工具（动态新增工具，无需修改config和核心代码）
        :param tool: 工具配置，格式与config.py中tools列表的工具格式一致
        :return: 注册成功返回True，失败返回False（工具名已存在则失败）
        """
        tool_name = tool["function"].get("name")
        if not tool_name:
            print(f"工具注册失败：工具名不能为空")
            return False
        if tool_name in self.tools:
            print(f"工具注册失败：工具【{tool_name}】已存在")
            return False
        self.tools[tool_name] = tool
        print(f"工具注册成功：工具【{tool_name}】")
        return True

    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """根据工具名查询工具配置，不存在则返回None"""
        return self.tools.get(tool_name)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """获取所有已注册的工具，返回列表格式（适配大模型工具调用参数）"""
        return list(self.tools.values())

    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        校验工具调用参数（避免参数缺失、格式错误）
        :param tool_name: 工具名
        :param parameters: 工具调用参数
        :return: 校验通过返回空字符串，失败返回错误信息
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return f"参数校验失败：工具【{tool_name}】不存在"

        # 获取工具的必填参数
        required_params = tool["function"]["parameters"].get("required", [])
        for param in required_params:
            if param not in parameters:
                return f"参数校验失败：工具【{tool_name}】缺少必传参数【{param}】"
        return ""

    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any], result: str) -> None:
        """记录工具调用日志（时间、工具名、参数、结果），便于排查问题"""
        import datetime
        log = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result
        }
        self.tool_call_logs.append(log)
        print(f"【工具调用日志】{log}")

    def clear_logs(self) -> None:
        """清空工具调用日志（避免日志过多占用内存）"""
        self.tool_call_logs.clear()

    def log_task_split(self, user_question: str, split_tasks: list) -> None:
        """
        记录任务拆解日志
        :param user_question: 用户原始复杂提问
        :param split_tasks: Agent拆解后的子任务列表
        """
        log = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_question": user_question,
            "split_tasks": split_tasks  # 拆解后的子任务列表
        }
        self.task_split_logs.append(log)
        print(f"✅ 任务拆解日志已记录：用户提问={user_question[:20]}...，拆解子任务{len(split_tasks)}个")


# 初始化工具管理器（全局单例，整个项目只需一个工具管理器实例）
tool_manager = ToolManager()

# 新增：动态注册翻译工具（使用register_tool方法）
# 翻译工具配置（格式与config.py中工具格式一致）
translate_tool = {
    "type": "function",
            "function": {
                "name": "get_city_code",
                "description": "根据城市名称，获取该城市的行政区划编码（如深圳→440300），用于后续天气查询等操作",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称（如深圳、广州、北京，仅支持国内城市）",

                        }
                    },
                    "required": ["city"] # 必传参数
                }
            }
}

# 调用register_tool方法，动态注册翻译工具
# tool_manager.register_tool(translate_tool)

# 测试工具管理器（可直接运行该文件，验证功能）
if __name__ == "__main__":
    # 测试1：查询所有工具
    all_tools = tool_manager.get_all_tools()
    print("=== 所有已注册工具 ===")
    for tool in all_tools:
        print(f"工具名：{tool['function']['name']}，描述：{tool['function']['description']}")



    # 测试3：参数校验
    test_params = {"location": "北京"}
    validate_result = tool_manager.validate_tool_parameters("search_current_weather", test_params)
    print(f"\n=== 参数校验结果 ===")
    print(validate_result if validate_result else "参数校验通过")

    # 测试4：记录工具调用日志
    tool_manager.log_tool_call("search_current_weather", test_params, "【北京实时天气】北京: 晴 18°C")
