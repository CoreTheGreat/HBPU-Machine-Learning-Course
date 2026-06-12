import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from typing import List, Dict, Any
from llm import LLMClient


class MemoryManager:
    def __init__(self, limit_turns: int = 5):
        """
        智能记忆管理器。
        
        :param limit_turns: 触发记忆整理与压缩的最大用户轮数阈值
        """
        self.system_prompt = ""
        # 短期对话上下文缓存（消息列表）
        self.history: List[Dict[str, Any]] = []
        
        # 长期记忆开发备忘录 (Development Memo)
        self.memo: str = "暂无开发记录。"
        self.limit_turns = limit_turns
        self.llm = None  # 延迟加载，防止相互引用

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def append(self, message: Any):
        """
        追加新消息到短期对话记忆。
        支持 dict 和 OpenAI Message 对象。
        """
        if isinstance(message, dict):
            self.history.append(message)
        else:
            # 兼容 OpenAI API 返回的 Message 对象
            msg_dict = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", ""),
            }
            # 处理 tool_calls
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            self.history.append(msg_dict)

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """
        融合 system_prompt、长期备忘录和短期会话，生成完整的对话列表。
        """
        full_system = (
            f"{self.system_prompt}\n\n"
            f"========================================\n"
            f"📋 【当前游戏开发备忘录 (已达成的设计决策)】\n"
            f"下面记录了之前步骤中已经确定的全局规范、代码架构和设计结论。\n"
            f"在后续开发中请务必严格遵守，维持开发的一致性：\n"
            f"{self.memo}\n"
            f"========================================"
        )
        
        messages = [{"role": "system", "content": full_system}]
        for msg in self.history:
            messages.append(msg)
        return messages

    def check_and_compress(self):
        """
        检查轮数。如果轮数超过限制，启动大模型总结，更新长期备忘录，并截断短期记忆。
        """
        user_turns = sum(1 for m in self.history if m.get("role") == "user")
        if user_turns <= self.limit_turns:
            return

        if not self.llm:
            self.llm = LLMClient()

        print("\n🧠 [记忆管理] 正在基于大模型压缩短期记忆并更新『长期开发备忘录』...")

        # 拼接近期的上下文信息
        summary_prompt = (
            f"请仔细阅读以下当前网页小游戏开发的历史会话记录。\n"
            f"请提取出其中所有已经确定的游戏设计规范、变量与逻辑接口名称、写好的 HTML/CSS/JS 文件名、"
            f"数值平衡公式及世界观背景等，将它们整合并更新到当前的『开发备忘录』中。\n\n"
            f"【原开发备忘录内容】：\n{self.memo}\n\n"
            f"【近期开发历史对话】：\n"
        )

        for msg in self.history:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "tool":
                summary_prompt += f"[工具执行反馈] {msg.get('name')} -> {content[:100]}...\n"
            elif role == "user":
                summary_prompt += f"[用户指令] -> {content[:200]}\n"
            else:
                summary_prompt += f"[助理思考] -> {content[:200]}\n"

        summary_prompt += (
            f"\n请输出一份全新、更新后的『网页游戏开发备忘录』结构化文本。"
            f"直接输出备忘录正文，无需任何前置或后置寒暄语。"
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": summary_prompt}]
            )
            updated_memo = response.choices[0].message.content
            if updated_memo:
                self.memo = updated_memo.strip()
                print("✅ [记忆管理] 长期开发备忘录已成功提炼并合并更新！")
                
                # 截断短期历史：仅保留最近 2 轮对话 (即最后 2 个 user 消息及之后的所有消息)
                user_indices = [i for i, m in enumerate(self.history) if m.get("role") == "user"]
                if len(user_indices) >= 2:
                    cutoff = user_indices[-2]
                    self.history = self.history[cutoff:]
                    print("🧹 [记忆管理] 短期会话缓存已安全截断，上下文环境已精简。")
        except Exception as e:
            print(f"⚠️ [记忆管理] 整理备忘录失败: {e}")

    def clear(self):
        """清空所有历史和备忘录"""
        self.history.clear()
        self.memo = "暂无开发记录。"


if __name__ == "__main__":
    print("--- 测试 MemoryManager 模块 ---")
    memory = MemoryManager(limit_turns=2)
    memory.set_system_prompt("你是一个开发助手。")
    memory.append({"role": "user", "content": "我想开发一个射击游戏，飞船大小是 40x40"})
    memory.append({"role": "assistant", "content": "好的，我已经记录了飞船大小为 40x40。"})
    
    # 模拟请求格式
    msgs = memory.get_messages_for_llm()
    print(f"合并后的系统 prompt:\n{msgs[0]['content']}")
    
    # 测试触发压缩
    memory.append({"role": "user", "content": "现在需要设计背景颜色为黑色"})
    memory.append({"role": "assistant", "content": "好的，背景为黑色。"})
    memory.append({"role": "user", "content": "把飞船的速度设为 5"})
    
    # 检测并执行压缩
    memory.check_and_compress()
    print(f"\n压缩后的备忘录:\n{memory.memo}")
    print(f"\n截断后的短期记忆条数: {len(memory.history)}")
