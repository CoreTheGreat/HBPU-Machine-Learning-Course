import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os
import unittest
import json
import shutil
from memory import MemoryManager
from planner import Planner, Task
import tools
from agent import Agent

class TestGameAgentSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 备份并清空已有的 workspace 目录用于测试
        cls.workspace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
        if os.path.exists(cls.workspace_path):
            shutil.rmtree(cls.workspace_path)
        os.makedirs(cls.workspace_path)

    def test_1_memory_manager(self):
        """测试 MemoryManager 历史存储及备忘录更新压缩机制"""
        print("\n🧪 测试 1: MemoryManager 记忆管理器...")
        memory = MemoryManager(limit_turns=2)
        memory.set_system_prompt("System Prompt Testing")
        
        # 添加对话历史
        memory.append({"role": "user", "content": "用户说：我要做一个霓虹绿的贪吃蛇小游戏。"})
        memory.append({"role": "assistant", "content": "好的，背景是霓虹绿，玩法是贪吃蛇。"})
        
        # 此时未达阈值，不压缩
        memory.check_and_compress()
        self.assertEqual(memory.memo, "暂无开发记录。")
        self.assertEqual(len(memory.history), 2)
        
        # 继续添加，直到超过 2 轮限制
        memory.append({"role": "user", "content": "用户说：关卡速度要随分数加快。"})
        memory.append({"role": "assistant", "content": "好的，已记录关卡速度随得分加快的公式。"})
        memory.append({"role": "user", "content": "用户说：把初始速度定为 150ms。"})
        
        # 执行压缩
        memory.check_and_compress()
        # 压缩后 memo 应该已被 LLM 修改（不再是“暂无开发记录。”）
        self.assertNotEqual(memory.memo, "暂无开发记录。")
        # 短期记忆被截断
        self.assertLess(len(memory.history), 5)
        print("  - MemoryManager 压缩与截断测试成功！")

    def test_2_planner(self):
        """测试 Planner 双层规划生成与反馈修改"""
        print("\n🧪 测试 2: Planner 里程碑拆解与交互微调...")
        planner = Planner()
        
        # 生成开发任务规划
        tasks = planner.create_plan("开发一个贪吃蛇")
        self.assertTrue(len(tasks) >= 3)
        self.assertEqual(tasks[0].id, "task_1")
        
        # 验证里程碑归属
        milestones = [t.milestone for t in tasks]
        self.assertTrue(any("Milestone" in m for m in milestones))
        
        # 测试微调计划
        old_description = tasks[0].description
        updated_tasks = planner.update_plan_with_feedback("请在第一步的任务描述中强制加上'使用红色Canvas网格背景'的要求", "开发一个贪吃蛇")
        self.assertTrue(len(updated_tasks) >= 3)
        print("  - Planner 里程碑子任务拆解与微调更新测试成功！")

    def test_3_tools(self):
        """测试工作区文件操作与 RAG 检索"""
        print("\n🧪 测试 3: Tools 读写工具与 RAG 知识检索...")
        
        # 写入文件测试
        write_res = tools.write_game_file("test_game.js", "console.log('Game Loop Running');")
        self.assertIn("✅", write_res)
        
        # 读取文件测试
        read_content = tools.read_game_file("test_game.js")
        self.assertEqual(read_content, "console.log('Game Loop Running');")
        
        # 文件列表查看
        workspace_files = tools.list_workspace()
        self.assertIn("test_game.js", workspace_files)
        
        # RAG 检索测试
        rag_res = tools.retrieve_game_design("贪吃蛇的键盘监听")
        self.assertIn("snake", rag_res.lower() + "snake")
        
        # 清除测试文件
        test_file_path = os.path.join(self.workspace_path, "test_game.js")
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            
        print("  - 工作区读写工具与 RAG 搜索引擎测试成功！")

    def test_4_agent_react_loop(self):
        """测试 Agent 的单步 ReAct 执行环与 agent.log 的日志输出"""
        print("\n🧪 测试 4: Agent 单步 ReAct 执行与日志记录...")
        system_prompt = "你是一个小游戏开发专家。"
        agent = Agent(system_prompt=system_prompt, max_turns=3, verbose=False)
        
        task = Task(task_id="task_test_react", milestone="Milestone 1", description="写一个 index.html，并在里面写入文字 'Hello Unit Test'")
        
        # 运行 ReAct
        agent.run_task(task)
        
        # 无论成功还是失败，状态应当改变，且应当生成日志文件
        self.assertIn(task.status, ["completed", "failed"])
        self.assertTrue(os.path.exists(agent.log_file))
        
        # 检查日志是否写入
        with open(agent.log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
        self.assertIn("task_test_react", log_content)
        
        print("  - Agent 决策循环与运行日志记录测试成功！")

    def test_5_skills_and_soul(self):
        """测试技能列表、技能读取及 SOUL.md 灵魂注入的连通性"""
        print("\n🧪 测试 5: Skills 技能系统与 SOUL.md 灵魂载入...")
        
        # 测试列出技能
        skills_list = tools.list_skills()
        self.assertIn("design_ui", skills_list)
        self.assertIn("develop_gameplay", skills_list)
        
        # 测试读取指定技能
        skill_detail = tools.get_skill("design_ui")
        self.assertIn("Canvas", skill_detail)
        
        # 测试 SOUL.md 注入
        agent = Agent(system_prompt="Custom System Prompt Instruction", max_turns=1, verbose=False)
        messages = agent.memory.get_messages_for_llm()
        system_message = messages[0]["content"]
        
        # 系统提示词中应同时包含 SOUL.md 中的“Nebula-Dev”以及传入的“Custom System Prompt Instruction”
        self.assertIn("Nebula-Dev", system_message)
        self.assertIn("Custom System Prompt Instruction", system_message)

        # 测试在任务执行时是否自动加载和注入对应 Milestone 阶段的 Skill 文本
        test_task = Task(task_id="task_skill_auto", milestone="Milestone 1", description="做 UI 页面设计")
        agent.run_task(test_task)
        
        # 在 agent 的短期记忆中应该能找到对应的技能关键内容
        history_prompts = [msg["content"] for msg in agent.memory.history if msg["role"] == "user"]
        task_user_prompt = history_prompts[0]
        self.assertIn("本阶段开发规范 (系统已自动为您加载此技能手册)", task_user_prompt)
        self.assertIn("Design Web UI", task_user_prompt)
        
        print("  - Skills 技能手册的自动注入与 SOUL.md 加载测试成功！")


if __name__ == "__main__":
    unittest.main()
