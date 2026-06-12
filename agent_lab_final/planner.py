import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import json
from typing import List, Dict, Any
from llm import LLMClient


class Task:
    def __init__(self, task_id: str, milestone: str, description: str, status: str = "pending", result: str = ""):
        """
        规划的单步子任务。
        
        :param task_id: 唯一任务标识 (如 task_1)
        :param milestone: 归属的标准里程碑名称 (如 Milestone 1)
        :param description: 具体子任务描述
        :param status: 当前状态 ('pending', 'running', 'completed', 'failed')
        :param result: 执行完毕后的产出或关联文件
        """
        self.id = task_id
        self.milestone = milestone
        self.description = description
        self.status = status
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "milestone": self.milestone,
            "description": self.description,
            "status": self.status,
            "result": self.result
        }


class Planner:
    def __init__(self):
        self.tasks: List[Task] = []
        self.llm = LLMClient()
        self.milestones = [
            "Milestone 1: 页面框架与 UI 设计 (HTML/CSS)",
            "Milestone 2: 核心玩法与渲染逻辑 (Canvas/JS)",
            "Milestone 3: 数值难度平衡与关卡配置 (RAG/JS)",
            "Milestone 4: 文件组装与打包连通性自测"
        ]

    def create_plan(self, user_requirements: str) -> List[Task]:
        """
        调用大模型根据里程碑与用户要求生成详细的开发计划。
        """
        system_prompt = (
            f"你是一个经验丰富的网页游戏项目经理。你的任务是将用户粗糙的游戏创意，\n"
            f"合理地细化并归入以下 4 个标准的里程碑（Milestone）中：\n"
            f"1. {self.milestones[0]}\n"
            f"2. {self.milestones[1]}\n"
            f"3. {self.milestones[2]}\n"
            f"4. {self.milestones[3]}\n\n"
            f"请为每个里程碑设计 1~2 个明确、无歧义的子任务（Subtask）。\n"
            f"你必须输出一个 JSON 格式的数组，其中包含多个子任务，每个子任务包含以下字段：\n"
            f"  - 'id': 任务 ID，例如 'task_1', 'task_2', ...\n"
            f"  - 'milestone': 归属的里程碑（必须是 'Milestone 1', 'Milestone 2', 'Milestone 3', 'Milestone 4' 之一）\n"
            f"  - 'description': 对该步骤要实现的具体功能和代码细节描述。\n\n"
            f"请严格输出纯 JSON 数组文本，不要包含 ```json 等标记，也不要输出任何多余的解释字样。"
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"我的开发创意是：{user_requirements}"}
                ]
            )
            raw_content = response.choices[0].message.content.strip()
            
            # 清理 Markdown 代码包裹标记
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            task_dicts = json.loads(raw_content)
            self.tasks = []
            for item in task_dicts:
                self.tasks.append(Task(
                    task_id=item["id"],
                    milestone=item["milestone"],
                    description=item["description"]
                ))
            return self.tasks
        except Exception as e:
            print(f"⚠️ [Planner] 大模型规划失败: {e}，正在生成默认骨架计划。")
            self.tasks = [
                Task("task_1", "Milestone 1", "创建开发区 index.html 并设计游戏面板布局与 CSS 样式"),
                Task("task_2", "Milestone 2", "编写主逻辑 main.js，实现键盘输入与核心玩法"),
                Task("task_3", "Milestone 2", "在 main.js 中加入碰撞检测逻辑与得分统计"),
                Task("task_4", "Milestone 3", "检索知识库中的数值平衡公式，调整蛇/游戏速度随得分的难度爬升曲线"),
                Task("task_5", "Milestone 4", "联调测试 HTML/JS 引用，确保打包无语法错误")
            ]
            return self.tasks

    def update_plan_with_feedback(self, feedback: str, user_requirements: str) -> List[Task]:
        """
        根据用户的修改建议，微调已生成的子任务计划列表。
        """
        current_plan_str = json.dumps([t.to_dict() for t in self.tasks], ensure_ascii=False, indent=2)
        
        prompt = (
            f"用户的初始需求是：{user_requirements}\n\n"
            f"【当前的开发规划】：\n{current_plan_str}\n\n"
            f"【用户提出的修改建议】：\n{feedback}\n\n"
            f"请根据用户的意见对当前的开发规划进行合理微调（包括重新编写 description、修改对应 Milestone 的步骤数等）。\n"
            f"你必须输出一个符合上述结构的 JSON 格式的数组。请直接输出纯 JSON，不要有 ```json 标签。"
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}]
            )
            raw_content = response.choices[0].message.content.strip()
            
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            task_dicts = json.loads(raw_content)
            self.tasks = []
            for item in task_dicts:
                self.tasks.append(Task(
                    task_id=item["id"],
                    milestone=item["milestone"],
                    description=item["description"]
                ))
            return self.tasks
        except Exception as e:
            print(f"⚠️ [Planner] 微调规划失败: {e}，保持原有计划。")
            return self.tasks


if __name__ == "__main__":
    print("--- 测试 Planner 模块 ---")
    planner = Planner()
    plan = planner.create_plan("开发一个黑客帝国风格的贪吃蛇")
    for t in plan:
        print(f"[{t.milestone}] {t.id}: {t.description}")
