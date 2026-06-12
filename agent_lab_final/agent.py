import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os
import json
import time
from typing import Callable, Optional
from llm import LLMClient
from tool_registry import tool_registry
from memory import MemoryManager
from planner import Planner, Task


class Agent:
    def __init__(self, system_prompt: str, max_turns: int = 8, verbose: bool = True):
        """
        网页游戏开发智能体控制器。
        
        :param system_prompt: 智能体的核心行为指令
        :param max_turns: 单个子任务的最大 ReAct 迭代步数
        :param verbose: 是否实时在标准输出打印详细思考日志
        """
        self.llm = LLMClient()
        self.max_turns = max_turns
        self.verbose = verbose
        
        # 动态加载 SOUL.md，融合注入智能体灵魂
        soul_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SOUL.md")
        soul_content = ""
        if os.path.exists(soul_path):
            try:
                with open(soul_path, "r", encoding="utf-8") as f:
                    soul_content = f.read().strip()
            except Exception as e:
                print(f"⚠️ 加载 SOUL.md 失败: {e}")

        full_prompt = system_prompt
        if soul_content:
            full_prompt = f"{soul_content}\n\n========================================\n{system_prompt}"
        
        # 初始化记忆管理器
        self.memory = MemoryManager()
        self.memory.set_system_prompt(full_prompt)
        
        # 初始化规划器
        self.planner = Planner()
        
        # 初始化日志文件
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.log")
        # 如果日志文件存在，清空旧日志
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except:
                pass
        
        self.log("🤖 网页游戏开发智能体框架初始化完成。")

    def log(self, message: str):
        """写日志并根据 verbose 状态输出到终端"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        if self.verbose:
            print(log_line)
            
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except:
            pass

    def run_task(self, task: Task) -> str:
        """
        以 ReAct 推理环模式，驱动工具反射执行并完成一个具体的子任务。
        
        :param task: 当前要执行的 Task 对象
        :return: 任务执行完成后的最终文本摘要或说明
        """
        task.status = "running"
        self.log(f"\n▶️ [任务开始] 执行任务 {task.id}: {task.description}")
        
        # 根据任务的里程碑阶段，自动读取并注入对应的开发技能指南 (Skills)
        skill_name = None
        if "Milestone 1" in task.milestone:
            skill_name = "design_ui"
        elif "Milestone 2" in task.milestone:
            skill_name = "develop_gameplay"
        elif "Milestone 3" in task.milestone:
            skill_name = "tune_difficulty"
        elif "Milestone 4" in task.milestone:
            skill_name = "verify_code"
            
        skill_content = ""
        if skill_name:
            skills_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
            skill_path = os.path.join(skills_dir, f"{skill_name}.md")
            if os.path.exists(skill_path):
                try:
                    with open(skill_path, "r", encoding="utf-8") as f:
                        skill_content = f.read().strip()
                    self.log(f"🧠 [自动读取技能] 针对 {task.milestone}，已成功载入技能指南: {skill_name}")
                except Exception as e:
                    self.log(f"⚠️ [自动读取技能] 读取技能 {skill_name} 失败: {e}")

        # 组装任务指派提示词，追加至短期对话上下文
        task_prompt = f"【当前任务目标】：{task.description}\n"
        if skill_content:
            task_prompt += (
                f"\n📋 【本阶段开发规范 (系统已自动为您加载此技能手册)】:\n"
                f"在开发本步骤的代码或逻辑时，请务必严格遵守以下技能规范：\n"
                f"{skill_content}\n"
                f"======================================\n"
            )
        task_prompt += (
            f"请在此步骤中完成该任务相关的代码开发、逻辑设计或文件维护。\n"
            f"你可以自主调用任何可用工具（可以单次调用多个工具）。\n"
            f"如果你的开发工作已全部完成或遇到了不可逾越的错误，请明确输出你的最终结论并终止本轮执行。"
        )
        self.memory.append({"role": "user", "content": task_prompt})
        
        turns = 0
        while turns < self.max_turns:
            # 整合 System Prompt + 长期备忘录 + 短期消息链
            messages = self.memory.get_messages_for_llm()
            tool_schemas = tool_registry.get_openai_tool_schemas()
            
            self.log(f"   [思考中] ReAct 第 {turns+1}/{self.max_turns} 轮迭代...")
            try:
                response = self.llm.chat(
                    messages=messages,
                    tools=tool_schemas if len(tool_schemas) > 0 else None
                )
            except Exception as e:
                err_msg = f"❌ 调用大模型 API 发生异常: {e}"
                self.log(err_msg)
                task.status = "failed"
                task.result = err_msg
                return err_msg

            assistant_message = response.choices[0].message
            self.memory.append(assistant_message)
            
            tool_calls = assistant_message.tool_calls
            
            # 情况 A: 模型确认开发完成，没有下发任何工具调用
            if not tool_calls:
                final_result = assistant_message.content or "无返回内容"
                self.log(f"✅ [任务完成] {task.id} 结束。结果摘要: {final_result[:150]}...")
                task.status = "completed"
                task.result = final_result
                
                # 触发短期记忆整理检测，将新阶段决策固化进备忘录，并截断多余历史
                self.memory.check_and_compress()
                return final_result
                
            # 情况 B: 模型要求调用工具，遍历反射执行
            self.log(f"   🔧 决定调用 {len(tool_calls)} 个工具:")
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                tool_call_id = tool_call.id
                
                try:
                    tool_args = json.loads(tool_args_str)
                except Exception as e:
                    tool_args = {}
                    self.log(f"   ⚠️ 参数解析失败: {tool_args_str} | 错误: {e}")
                
                self.log(f"     -> 工具: {tool_name} | 参数: {tool_args}")
                
                # 反射执行工具并写入反馈
                tool_result = tool_registry.execute(tool_name, tool_args)
                result_str = str(tool_result)
                
                self.log(f"     <- 结果: {result_str[:150]}...")
                
                self.memory.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result_str
                })
                
            turns += 1
            
        err_msg = f"❌ 任务运行失败：超过了单步最大轮数 {self.max_turns}。"
        self.log(err_msg)
        task.status = "failed"
        task.result = err_msg
        return err_msg

    def execute_all_tasks(self, callback: Optional[Callable[[Task], None]] = None):
        """
        循环迭代，自动顺序执行所有已规划的任务。
        """
        self.log("🚀 开始顺序执行任务清单（Plan-and-Execute 双环运行中）...")
        for task in self.planner.tasks:
            if task.status in ["completed", "running"]:
                continue
                
            if callback:
                # 触发 UI 刷新回调
                callback(task)
                
            self.run_task(task)
            
            if task.status == "failed":
                self.log(f"🚨 任务 {task.id} 失败，执行中断！请检查代码或配置。")
                break
                
        self.log("🏁 任务规划队列执行结束。")
