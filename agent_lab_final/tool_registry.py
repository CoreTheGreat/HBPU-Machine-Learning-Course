import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import inspect
import json
import re
from typing import get_type_hints, Callable, Any


class ToolRegistry:
    def __init__(self):
        # 存储注册的工具函数：name -> function
        self.tools = {}
        # 存储对应的 OpenAI Tool Schema：name -> schema_dict
        self.schemas = {}

    def register(self, func: Callable) -> Callable:
        """
        注册工具函数的装饰器/方法。
        """
        name = func.__name__
        self.tools[name] = func
        self.schemas[name] = self._generate_schema(func)
        return func

    def _generate_schema(self, func: Callable) -> dict:
        """
        通过反射、类型提示和 docstring 自动生成 OpenAI 规范的 Tool Schema。
        """
        sig = inspect.signature(func)
        doc = func.__doc__ or ""
        
        # 1. 解析 docstring，提取函数描述和参数描述
        func_desc, param_descs = self._parse_docstring(doc)

        properties = {}
        required = []

        # 2. 获取类型提示
        type_hints = get_type_hints(func)

        # 3. 遍历参数生成 properties
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # 映射 Python 类型到 JSON Schema 类型
            py_type = type_hints.get(param_name, str)
            json_type = self._map_python_type_to_json(py_type)

            param_desc = param_descs.get(param_name, f"参数 {param_name}")

            properties[param_name] = {
                "type": json_type,
                "description": param_desc
            }

            # 如果没有默认值，则是必填参数
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # 4. 构建符合 OpenAI 要求的 Function 格式
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_desc,
                "parameters": {
                    "type": "object",
                    "properties": properties
                }
            }
        }
        if required:
            schema["function"]["parameters"]["required"] = required

        return schema

    def _parse_docstring(self, docstring: str) -> tuple[str, dict[str, str]]:
        """
        简易 Docstring 解析器，支持 Sphinx 风格 (:param name: desc)
        和 Google 风格 (Args: \n name: desc)。
        """
        if not docstring:
            return "没有提供描述。", {}

        lines = [line.strip() for line in docstring.strip().split("\n")]
        # 函数描述一般为第一行
        func_desc = lines[0] if lines else "没有提供描述。"

        param_descs = {}
        # Sphinx 风格匹配
        sphinx_pattern = re.compile(r"^:param\s+(\w+):\s*(.*)$")
        # Google 风格匹配
        google_pattern = re.compile(r"^(\w+)\s*:\s*(.*)$")

        in_args_section = False
        for line in lines:
            # Sphinx 匹配
            sphinx_match = sphinx_pattern.match(line)
            if sphinx_match:
                name, desc = sphinx_match.groups()
                param_descs[name] = desc
                continue

            # Google 风格匹配 (遇到 Args: 或 Parameters: 开始)
            if line.lower() in ("args:", "parameters:", "arguments:"):
                in_args_section = True
                continue

            if in_args_section:
                google_match = google_pattern.match(line)
                if google_match:
                    name, desc = google_match.groups()
                    param_descs[name] = desc

        return func_desc, param_descs

    def _map_python_type_to_json(self, py_type: Any) -> str:
        """
        映射 Python 常用类型到 JSON Schema 类型。
        """
        if py_type == int:
            return "integer"
        elif py_type == float:
            return "number"
        elif py_type == bool:
            return "boolean"
        elif py_type == list or getattr(py_type, "__origin__", None) == list:
            return "array"
        elif py_type == dict or getattr(py_type, "__origin__", None) == dict:
            return "object"
        else:
            return "string"

    def execute(self, name: str, arguments: dict) -> Any:
        """
        执行已注册的工具。
        
        :param name: 工具名称
        :param arguments: 大模型输出的参数字典
        """
        if name not in self.tools:
            return f"错误: 未找到名为 '{name}' 的工具。"
        
        func = self.tools[name]
        
        try:
            # 这里的 arguments 已经是解析好的 dict，直接进行解包调用
            result = func(**arguments)
            return result
        except Exception as e:
            return f"执行工具 '{name}' 时出错: {str(e)}"

    def get_openai_tool_schemas(self) -> list[dict]:
        """
        获取注册的所有工具 Schema 列表。
        """
        return list(self.schemas.values())


# 全局注册表实例
tool_registry = ToolRegistry()

# 装饰器
def tool(func: Callable) -> Callable:
    """
    注册工具函数的装饰器：
    @tool
    def my_func(a: int):
        ...
    """
    return tool_registry.register(func)


if __name__ == "__main__":
    print("--- 测试 ToolRegistry 模块 ---")
    
    # 定义测试工具函数
    @tool
    def add_numbers(x: int, y: int, label: str = "sum") -> str:
        """
        计算两个数字之和并返回带有标签的字符串结果。
        
        :param x: 第一个操作数
        :param y: 第二个操作数
        :param label: 输出的前缀标签
        """
        return f"{label}: {x + y}"
    
    # 1. 验证 Schema 生成是否正确
    print("\n1. [验证] 自动生成工具 Schema (JSON Format):")
    schemas = tool_registry.get_openai_tool_schemas()
    print(json.dumps(schemas, indent=2, ensure_ascii=False))
    
    # 2. 验证反射调用是否正常
    print("\n2. [验证] 反射动态执行工具 (add_numbers, x=5, y=15):")
    result = tool_registry.execute("add_numbers", {"x": 5, "y": 15})
    print(f"执行结果 -> {result}")
