# utils/memory_manager.py
from typing import List, Dict, Any
import json
import os
from datetime import datetime


class MemoryManager:
    """Agent记忆管理类：实现短期记忆（对话上下文）和长期记忆（本地存储）的管理"""

    def __init__(self):
        # 1. 短期记忆：存储当前对话的上下文（用户提问、工具调用、回答）
        self.short_term_memory: List[Dict[str, Any]] = []
        # 2. 长期记忆：存储跨对话的关键信息，从本地文件加载（若文件不存在则初始化）
        self.long_term_memory: List[Dict[str, Any]] = []
        # 长期记忆存储路径（项目根目录下的memory_store目录）
        self.long_term_memory_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # 定位到项目根目录
            "memory_store", "history_memory.json"
        )

        # 初始化：创建存储目录、加载长期记忆
        self._init_long_term_memory()

    def _init_long_term_memory(self) -> None:
        """初始化长期记忆：创建存储目录，加载本地记忆文件"""
        # 创建memory_store目录（若不存在）
        store_dir = os.path.dirname(self.long_term_memory_path)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        # 加载本地记忆文件（若不存在则创建空文件）
        if os.path.exists(self.long_term_memory_path):
            try:
                with open(self.long_term_memory_path, "r", encoding="utf-8") as f:
                    self.long_term_memory = json.load(f)
                print(f"✅ 成功加载长期记忆，共{len(self.long_term_memory)}条历史记录")
            except json.JSONDecodeError:
                print("⚠️  长期记忆文件格式错误，重新初始化")
                self.long_term_memory = []
                self.save_long_term_memory()
        else:
            # 新建空的长期记忆文件
            self.save_long_term_memory()
            print("✅ 长期记忆文件初始化完成")

    def add_short_term_memory(self, content: Dict[str, Any]) -> None:
        """
        新增短期记忆（当前对话上下文）
        :param content: 记忆内容，格式：{"role": "user/tool/assistant", "content": "内容", "time": "时间"}
        """
        # 补充时间戳，便于追溯
        content["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.short_term_memory.append(content)
        # 限制短期记忆长度（最多存储10条，避免内存占用过多）
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)  # 删除最早的一条记忆

    def get_short_term_memory(self) -> List[Dict[str, Any]]:
        """获取当前对话的短期记忆（上下文）"""
        return self.short_term_memory

    def clear_short_term_memory(self) -> None:
        """清空短期记忆（切换对话时使用）"""
        self.short_term_memory.clear()
        print("✅ 短期记忆已清空")

    def add_long_term_memory(self, content: Dict[str, Any]) -> None:
        """
        新增长期记忆（跨对话保存）
        :param content: 记忆内容，格式：{"type": "tool_call/user_preference", "content": "内容", "time": "时间"}
        """
        content["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.long_term_memory.append(content)
        if len(self.short_term_memory) > 50:
            self.short_term_memory.pop(0)  # 删除最早的一条记忆
        # 保存到本地文件（实时持久化）
        self.save_long_term_memory()
        print(f"✅ 新增长期记忆：{content['content'][:20]}...")

    def get_long_term_memory(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取长期记忆（默认获取最近5条，避免信息过多干扰大模型）
        :param limit: 限制获取条数
        """
        return self.long_term_memory[-limit:] if len(self.long_term_memory) > limit else self.long_term_memory

    def save_long_term_memory(self) -> None:
        """将长期记忆保存到本地JSON文件（持久化存储）"""
        try:
            with open(self.long_term_memory_path, "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"⚠️  长期记忆保存失败：{str(e)[:50]}")

    def clear_long_term_memory(self) -> None:
        """清空长期记忆（谨慎使用）"""
        self.long_term_memory = []
        self.save_long_term_memory()
        print("✅ 长期记忆已清空（本地文件同步更新）")



# 初始化记忆管理器（全局单例，与工具管理器对应，整个项目只需一个实例）
memory_manager = MemoryManager()

# 测试记忆管理器（可直接运行该文件，验证功能）
if __name__ == "__main__":
    # 测试1：新增短期记忆
    memory_manager.add_short_term_memory({
        "role": "user",
        "content": "查询北京天气"
    })
    memory_manager.add_short_term_memory({
        "role": "tool",
        "content": "【北京实时天气】北京: 晴 18°C"
    })
    print("=== 短期记忆 ===")
    for mem in memory_manager.get_short_term_memory():
        print(f"{mem['time']} | {mem['role']}: {mem['content']}")

    # 测试2：新增长期记忆
    memory_manager.add_long_term_memory({
        "type": "tool_call",
        "content": "用户曾调用search_current_weather工具，查询北京天气"
    })
    memory_manager.add_long_term_memory({
        "type": "user_preference",
        "content": "用户偏好使用zhipu模型"
    })
    print("\n=== 长期记忆（最近2条） ===")
    for mem in memory_manager.get_long_term_memory(limit=2):
        print(f"{mem['time']} | {mem['type']}: {mem['content']}")

    # 测试3：清空短期记忆
    memory_manager.clear_short_term_memory()
    print(f"\n=== 清空后短期记忆 ===")
    print(memory_manager.get_short_term_memory())