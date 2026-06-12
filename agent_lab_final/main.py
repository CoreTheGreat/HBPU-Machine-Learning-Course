import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os
from agent import Agent
import tools  # 自动注册文件操作与本地服务工具
from tool_registry import tool_registry

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.live import Live
    from rich.status import Status
    from rich.markdown import Markdown
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


def welcome_banner():
    title = "🎮 HBPU-AI Web Game Development Agent (Final Version)"
    subtitle = "基于 Plan-and-Execute 双环规划、局部记忆压缩与 Web 容器托管的自制游戏引擎"
    if HAS_RICH:
        panel = Panel(
            Text(f"{title}\n{subtitle}", justify="center", style="bold cyan"),
            border_style="cyan",
            expand=False
        )
        console.print(panel)
        console.print("💡 [bold green]请输入您想要开发的小游戏创意（例如：“一个赛博朋克风的贪吃蛇小游戏”）：[/bold green]")
        console.print("💡 [bold yellow]特殊指令: /exit (退出), /clear (重置工作区和记忆), /tools (可用工具), /log (查看运行日志)[/bold yellow]\n")
    else:
        print("=" * 70)
        print(title)
        print(subtitle)
        print("=" * 70)
        print("💡 请输入您想要开发的小游戏创意：")
        print("💡 特殊指令: /exit (退出), /clear (重置工作区和记忆), /tools (可用工具), /log (查看运行日志)\n")


def make_tasks_table(tasks) -> Table:
    """创建并返回开发规划任务的表格视图"""
    table = Table(title="📋 网页游戏开发子任务清单", show_header=True, header_style="bold magenta")
    table.add_column("任务 ID", style="dim", width=8)
    table.add_column("里程碑阶段", style="cyan", width=26)
    table.add_column("子任务执行目标", style="white", width=50)
    table.add_column("执行状态", justify="right", width=12)
    
    for task in tasks:
        status_str = "[grey]⚪ 等待中[/grey]"
        if task.status == "running":
            status_str = "[bold blue]🔵 执行中...[/bold blue]"
        elif task.status == "completed":
            status_str = "[bold green]🟢 已完成[/bold green]"
        elif task.status == "failed":
            status_str = "[bold red]🔴 失败[/bold red]"
            
        table.add_row(task.id, task.milestone, task.description, status_str)
    return table


def run_interactive_cli():
    welcome_banner()
    
    system_prompt = (
        "你是一个顶尖的网页前端小游戏专家兼高级架构师。\n"
        "你能够独立通过 HTML5 Canvas、CSS3 与原生 Javascript 完成各种高可玩性、高视觉观感的 2D 网页游戏。\n"
        "【开发原则与规范】\n"
        "1. 一切页面资源和核心代码必须生成并存放在 workspace/ 目录中。\n"
        "2. index.html 文件必须包含完整的游戏骨架和引入对应的 CSS/JS 文件。\n"
        "3. 在处理复杂动画（如游戏主循环 Game Loop）、输入监听或难度变迁公式时，\n"
        "   必须主动调用 retrieve_game_design 检索本地知识库，严禁使用过时或错误的 JS 写法。\n"
        "4. 你拥有一个完备的技能库（skills/），包含了设计 UI (design_ui)、实现核心玩法 (develop_gameplay)、\n"
        "   数值调平 (tune_difficulty) 以及联调校验 (verify_code) 的技能。在执行各个 Milestone 阶段的子任务时，\n"
        "   必须先主动调用 list_skills 和 get_skill 读取对应技能的指令与规约，并严格按照技能描述操作。\n"
        "5. 写完文件后，主动通过 list_workspace 查看文件生成状态，并启动游戏本地服务器 start_game_server 方便用户测试。"
    )
    
    try:
        agent = Agent(system_prompt=system_prompt, max_turns=10, verbose=False)
    except Exception as e:
        print(f"❌ 智能体系统启动失败: {e}")
        return

    while True:
        try:
            if HAS_RICH:
                user_input = console.input("[bold green]👤 User > [/bold green]").strip()
            else:
                user_input = input("👤 User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n✔ 退出程序。再见！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd == "/exit":
            print("✔ 退出程序。再见！")
            break
        elif cmd == "/clear":
            agent.memory.clear()
            agent.planner.tasks.clear()
            # 清理工作区文件
            workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
            if os.path.exists(workspace_dir):
                for f in os.listdir(workspace_dir):
                    try:
                        os.remove(os.path.join(workspace_dir, f))
                    except:
                        pass
            if HAS_RICH:
                console.print("[bold green]✔ 记忆已重置，游戏工作区已清空。[/bold green]")
            else:
                print("✔ 记忆已重置，游戏工作区已清空。")
            continue
        elif cmd == "/tools":
            schemas = tool_registry.get_openai_tool_schemas()
            print("\n🔧 已向大模型注册的工具库：")
            for schema in schemas:
                func = schema["function"]
                print(f"  • {func['name']}: {func['description']}")
            print()
            continue
        elif cmd == "/log":
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.log")
            if not os.path.exists(log_path):
                print("📝 暂无运行日志。")
            else:
                print("\n📝 --- 智能体运行日志 (agent.log) ---")
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    # 仅打印最后 40 行日志以防爆屏
                    for line in lines[-40:]:
                        print(line, end="")
                except Exception as e:
                    print(f"读取日志失败: {e}")
                print("-----------------------------------\n")
            continue

        # ----------------------------------------------------
        # 1. 规划阶段：双层 Milestones + LLM 拆解
        # ----------------------------------------------------
        if HAS_RICH:
            with Status("[cyan]Planner 正在分析创意并定制开发计划...", console=console) as status:
                tasks = agent.planner.create_plan(user_input)
        else:
            print("Planner 正在生成开发计划...")
            tasks = agent.planner.create_plan(user_input)

        # ----------------------------------------------------
        # 2. 交互式确认阶段
        # ----------------------------------------------------
        while True:
            if HAS_RICH:
                console.print(make_tasks_table(tasks))
                console.print("\n💡 [bold yellow]请确认当前游戏开发清单是否符合预期？[/bold yellow]")
                console.print("💡 [bold green]输入 'y' 启动自动编码开发；输入具体的意见让大模型修改规划。[/bold green]")
                feedback = console.input("[bold cyan]👉 确认/修改 > [/bold cyan]").strip()
            else:
                for t in tasks:
                    print(f"[{t.milestone}] {t.id}: {t.description} (status: {t.status})")
                feedback = input("👉 确认/修改 (y/意见) > ").strip()

            if not feedback:
                continue

            if feedback.lower() in ["y", "yes", "ok"]:
                break
            else:
                # 重新调整规划
                if HAS_RICH:
                    with Status("[cyan]Planner 正在根据您的反馈更新开发策略...", console=console) as status:
                        tasks = agent.planner.update_plan_with_feedback(feedback, user_input)
                else:
                    print("Planner 正在更新开发策略...")
                    tasks = agent.planner.update_plan_with_feedback(feedback, user_input)

        # ----------------------------------------------------
        # 3. 执行阶段：ReAct 闭环编码与文件生成
        # ----------------------------------------------------
        if HAS_RICH:
            console.print("\n🚀 [bold green]已确认规划，Agent 正在启动 ReAct 双环编码流，请稍后...[/bold green]\n")
            with Live(make_tasks_table(agent.planner.tasks), console=console, refresh_per_second=2) as live:
                def update_ui(task):
                    live.update(make_tasks_table(agent.planner.tasks))
                
                agent.execute_all_tasks(callback=update_ui)
                live.update(make_tasks_table(agent.planner.tasks))
        else:
            print("\n已确认规划，开始执行任务序列...")
            agent.execute_all_tasks()

        # ----------------------------------------------------
        # 4. 后处理与测试挂接
        # ----------------------------------------------------
        # 调用工具启动本地托管服务器，方便用户玩
        if HAS_RICH:
            console.print("\n[bold cyan]🔧 正在完成整合与部署本地 HTTP 容器...[/bold cyan]")
        else:
            print("\n正在启动本地测试服务器...")
            
        server_result = tools.start_game_server()
        
        if HAS_RICH:
            console.print(Panel(Markdown(server_result), title="🎮 游戏发布状态", border_style="cyan"))
        else:
            print("\n=== 游戏发布状态 ===")
            print(server_result)
            print("===================\n")


if __name__ == "__main__":
    run_interactive_cli()
