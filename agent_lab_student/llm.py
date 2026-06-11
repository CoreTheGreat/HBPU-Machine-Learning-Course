import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import httpx
from openai import OpenAI
from config import config

class LLMClient:
    def __init__(self):
        # 确保配置已加载且有效
        config.validate()
        
        # 💡 如果在 Windows 下使用 VPN/代理遇到 SSL 报错，可以用不验证 SSL 的 Client:
        # self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, http_client=httpx.Client(verify=False))
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model_name = config.model_name

    def chat(self, messages: list, tools: list = None, stream: bool = False):
        """
        发送聊天请求到大模型。
        
        :param messages: 对话历史消息列表，每个元素形如 {'role': 'user', 'content': '...'}
        :param tools: 大模型可选调用的工具定义列表（OpenAI Tool Schema 格式）
        :param stream: 是否启用流式输出（注意：Function Calling 建议先使用非流式以简化解析逻辑）
        :return: OpenAI ChatCompletion 响应对象
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # 仅在提供了工具且工具列表不为空时传入 tools 参数
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            print(f"❌ 调用大模型 API 发生异常: {e}")
            raise e


if __name__ == "__main__":
    print("--- 测试 LLMClient ---")
    try:
        client = LLMClient()
        print(f"✅ LLM 客户端初始化成功！使用模型: {client.model_name}")
        
        # 尝试发送简短的测试请求
        print("正在尝试向大模型发送测试消息...")
        response = client.chat(messages=[{"role": "user", "content": "你好，请用两个字回答：收到。"}])
        print(f"大模型响应结果: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ 运行测试时遇到问题: {e}")
        print("请检查当前目录下是否配置了正确的 `config.json` 或设置了 API 环境变量。")
