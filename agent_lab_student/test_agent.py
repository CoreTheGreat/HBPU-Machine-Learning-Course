import unittest
import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from tool_registry import ToolRegistry, tool, tool_registry
from agent import Agent
from config import config

# 必须加载且验证有效的真实 API 密钥
config.validate()


class TestAgentFramework(unittest.TestCase):

    def setUp(self):
        # 每次测试前，将全局工具注册表重置/清空，避免测试用例相互干扰
        tool_registry.tools.clear()
        tool_registry.schemas.clear()

    def test_tool_registration(self):
        """
        测试用例 1：验证 @tool 装饰器是否能成功将工具注册到 ToolRegistry 中，并且生成正确的 Schema
        """
        @tool
        def mock_tool_abc(param_a: int, param_b: str = "default_val") -> str:
            """
            这是一个用于单元测试的模拟工具函数。
            
            :param param_a: 第一个整型参数说明
            :param param_b: 第二个字符型参数说明
            """
            return f"Result: {param_a} - {param_b}"

        # 检查是否成功注册到字典中
        self.assertIn("mock_tool_abc", tool_registry.tools)
        self.assertIn("mock_tool_abc", tool_registry.schemas)

        # 检查自动生成的 JSON Schema 格式
        schema = tool_registry.schemas["mock_tool_abc"]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "mock_tool_abc")
        self.assertEqual(schema["function"]["description"], "这是一个用于单元测试的模拟工具函数。")
        
        # 验证必需参数 (没有默认值的 param_a 应该是必填项)
        required_params = schema["function"]["parameters"].get("required", [])
        self.assertIn("param_a", required_params)
        self.assertNotIn("param_b", required_params)

    def test_tool_execution(self):
        """
        测试用例 2：验证 ToolRegistry 的 execute 动态反射调用机制
        """
        @tool
        def multiply(x: int, y: int) -> int:
            """
            计算乘积。
            
            :param x: 乘数
            :param y: 被乘数
            """
            return x * y

        # 正常反射执行调用测试
        res = tool_registry.execute("multiply", {"x": 6, "y": 7})
        self.assertEqual(res, 42)

        # 异常调用测试：未注册的工具
        error_res = tool_registry.execute("non_exist_tool", {})
        self.assertTrue(error_res.startswith("错误"))

    def test_agent_run_loop(self):
        """
        测试用例 3：运行一次真实的 ReAct 决策循环，验证 Agent 控制流与 LLM 的连通性。
        这个测试需要正确的 config.json 以及大模型 API 的支持。
        """
        @tool
        def calculate(expression: str) -> str:
            """
            一个高精度数学表达式计算器。支持加减乘除、括号等运算。
            
            :param expression: 需要计算的数学表达式字符串，如 '200000 - 5000 - 60000'
            """
            try:
                clean_expr = expression.replace(" ", "")
                if not all(c in "0123456789+-*/()." for c in clean_expr):
                    return "错误: 表达式包含非法字符。只能包含数字和 +-*/()."
                result = eval(clean_expr, {"__builtins__": None}, {})
                return f"{expression} = {result}"
            except Exception as e:
                return f"计算出错: {str(e)}"

        # 实例化一个具有简短系统提示词的 Agent
        test_agent = Agent(system_prompt="你是一个极简助手。当需要计算时请调用计算器工具。", verbose=True)

        # 提出一个需要进行计算的问题
        question = "计算 125 乘以 8 等于多少？"
        print(f"\n[运行单元测试中...] 发送问题: '{question}'")
        
        reply = test_agent.run(question)
        print(f"[真实测试回复] -> {reply}\n")

        # 校验：答复中应该包含 1000
        self.assertIn("1000", reply)
        
        # 校验：检查 memory 链条中是否保留了 'tool' 角色的对话记录
        has_tool_message = any(msg.get("role") == "tool" for msg in test_agent.memory if isinstance(msg, dict))
        self.assertTrue(has_tool_message, "Agent 的对话历史记忆中没有包含 tool 执行的返回记录！")


if __name__ == '__main__':
    unittest.main()
