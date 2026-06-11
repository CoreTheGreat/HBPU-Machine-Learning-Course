import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import json
from llm import LLMClient
from tool_registry import tool_registry

class Agent:
    def __init__(self, system_prompt: str = "你是一个全能的智能助手。", max_turns: int = 10, verbose: bool = True):
        """
        初始化智能体。
        
        :param system_prompt: 系统提示词，用于设定智能体的人设和行为准则
        :param max_turns: 最大工具调用交互轮数，防止陷入死循环
        :param verbose: 是否在终端打印详细的工具调用思考过程
        """
        self.llm = LLMClient()
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.verbose = verbose
        
        # 对话历史，首位预置 System Prompt
        self.memory = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_message: str) -> str:
        """
        核心智能体执行循环。
        
        :param user_message: 用户输入的文本指令
        :return: 智能体的最终文本答复
        """
        # ==========================================
        # 🎯 TODO: 学生需要实现核心的 ReAct 推理与工具执行循环逻辑：
        #
        # 步骤 1: 将用户的输入（user_message）以 dict 格式追加到 self.memory 对话历史中（角色为 'user'）
        #
        # 步骤 2: 启动循环，控制最大交互轮数在 self.max_turns 内：
        #        a) 从 tool_registry 中获取当前已注册的所有工具的 OpenAI Schemas。
        #        b) 调用 self.llm.chat(...) 发送当前的对话历史（self.memory）和大模型可选的 tools 列表。
        #        c) 提取大模型回复的 Message 对象 (assistant_message = response.choices[0].message)。
        #        d) ⚠️重要：必须将该助理的 Message 原始对象直接追加（append）到 self.memory 中（包含 tool_calls 的消息也必须先存入历史）。
        #        e) 判断大模型回复中是否包含了工具调用请求 (tool_calls = assistant_message.tool_calls)。
        #
        # 🟢 分支 A (没有工具调用):
        #        如果没有 tool_calls，说明大模型推理完成并给出了最终文本答复。直接返回 assistant_message.content 并退出循环。
        #
        # 🔵 分支 B (需要工具调用):
        #        遍历 tool_calls 中的每一个工具调用（一个 assistant_message 可能包含多个并发工具调用）：
        #           i)   解析出工具名 (tool_name)、参数 (tool_args_str) 以及调用 ID (tool_call_id)。
        #           ii)  解析 JSON 参数字符串为 Python 字典格式（可以使用 json.loads(tool_args_str)）。
        #           iii) 调用 tool_registry.execute(tool_name, tool_args) 动态反射执行该 Python 函数，获取返回结果（转化为 string）。
        #           iv)  ⚠️重要：将工具执行结果填回对话历史。构建一个 dict 追加到 self.memory，其格式必须为:
        #                {
        #                    "role": "tool",
        #                    "tool_call_id": tool_call_id,
        #                    "name": tool_name,
        #                    "content": 工具执行结果字符串
        #                }
        #        f) 轮数 turns 加 1，继续下一轮循环大模型推理。
        #
        # 步骤 3: 兜底处理。如果循环超过了最大轮数依然没有输出最终回答，返回错误描述并追加到 memory。
        # ==========================================
        pass

    def clear_memory(self):
        """
        清空对话历史（保留 System Prompt）。
        """
        self.memory = [{"role": "system", "content": self.system_prompt}]


if __name__ == "__main__":
    print("--- 测试 Agent 模块 ---")
    try:
        agent = Agent(system_prompt="你是一个简单的测试助手。", verbose=True)
        print("✅ Agent 实例化及 LLMClient 连接初始化成功！")
        print(f"初始对话上下文: {agent.memory}")
    except Exception as e:
        print(f"❌ 初始化 Agent 失败: {e}")
        print("请确保已在当前文件夹下提供了可用的 `config.json` 或配置了 API Key 环境变量。")
