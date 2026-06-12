import os
import sys
import json
from pathlib import Path
from openai import OpenAI

# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class Config:
    def __init__(self):
        self.api_key = None
        self.base_url = None
        self.model_name = None
        self._load_config()

    def _load_config(self):
        # 1. 只从当前脚本所在目录的 config.json 加载
        config_path = Path(__file__).parent / "config.json"

        config_data = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"警告: 读取配置文件 {config_path} 失败: {e}")

        # 2. 从配置文件中读取（支持多 API 模式与单 API 模式）
        active_provider = config_data.get("active_provider")
        providers = config_data.get("providers", {})

        if active_provider and active_provider in providers:
            # 多 API 模式
            provider_config = providers[active_provider]
            self.api_key = provider_config.get("api_key")
            self.base_url = provider_config.get("base_url")
            self.model_name = provider_config.get("model_name")
        else:
            # 兼容原有的单 API 模式
            self.api_key = config_data.get("api_key")
            self.base_url = config_data.get("base_url")
            self.model_name = config_data.get("model_name")

    def validate(self):
        if not self.api_key:
            raise ValueError(
                "❌ 未找到 API Key！\n"
                "请确保当前目录下的 `config.json` 文件存在、配置正确，且填写了有效的 API 密钥。"
            )
        return True

    def select_provider(self, provider_name: str, save_to_file: bool = False):
        """
        动态切换当前激活的大模型 API 提供商。
        
        :param provider_name: 提供商名称，例如 'agnes', 'siliconflow', 'deepseek', 'mino', 'kimi'
        :param save_to_file: 是否将修改同步保存写入 config.json 配置文件中
        """
        config_path = Path(__file__).parent / "config.json"
        
        if not config_path.exists():
            raise ValueError("❌ 未找到配置文件 config.json，无法切换提供商。")
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ 读取配置文件失败: {e}")
            
        providers = config_data.get("providers", {})
        if provider_name not in providers:
            raise ValueError(f"❌ 提供商 '{provider_name}' 不存在于配置文件中。可选的提供商为: {list(providers.keys())}")
            
        # 1. 切换内存中的参数
        provider_config = providers[provider_name]
        self.api_key = provider_config.get("api_key")
        self.base_url = provider_config.get("base_url")
        self.model_name = provider_config.get("model_name")
        
        # 2. 如果要求保存，写入配置文件
        if save_to_file:
            config_data["active_provider"] = provider_name
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print(f"✅ 已成功切换激活 API 提供商为 '{provider_name}'，并且已同步写入 config.json！")
            except Exception as e:
                print(f"⚠️ 切换内存成功，但写入配置文件失败: {e}")
        else:
            print(f"✅ 已成功在内存中动态切换激活 API 提供商为 '{provider_name}'。")

# 单例模式，全局共享配置
config = Config()

if __name__ == "__main__":
    print("--- 测试 Config 模块 ---")
    
    # 打印当前加载的所有 API 提供商列表
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                active = data.get("active_provider", "未设定")
                providers = data.get("providers", {})
                print(f"📖 配置文件中共加载了 {len(providers)} 个 API 提供商：")
                for name, p_info in providers.items():
                    status = "🌟【当前激活】" if name == active else ""
                    print(f"  • {name:<12} {status}")
                    print(f"    - URL: {p_info.get('base_url')}")
                    print(f"    - Model: {p_info.get('model_name')}")
        except Exception as e:
            print(f"读取配置列表出错: {e}")
            
    print("\n💡 你可以输入上面列表中的提供商名称进行动态切换并测试连接，直接回车将测试当前激活的提供商。")
    user_choice = input("请输入提供商名称 (或直接回车): ").strip()
    if user_choice:
        try:
            config.select_provider(user_choice, save_to_file=True)
        except Exception as e:
            print(e)
            sys.exit(1)
            
    print(f"\n🔄 正在连接测试提供商...")
    print(f"  - API Key (部分): {config.api_key[:12] + '...' if config.api_key else '未配置'}")
    print(f"  - Base URL: {config.base_url}")
    print(f"  - Model Name: {config.model_name}")
    
    try:
        config.validate()
        print("✅ 配置格式验证成功！")
        
        # 发送真实的轻量对话请求以验证 API 密钥和网络连通性
        print("正在尝试向大模型 API 发送测试对话...")
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "你好，请回答两个字'正常'"}]
        )
        print(f"✅ API 对话测试成功！收到回复: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ API 连通性测试失败: {e}")


