import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from agent import Agent
import tools # 自动导入并注册 tools 中的业务工具



# ==========================================
# 🎨 2. 交互式对话 CLI 循环实现
# ==========================================

# 尝试导入 rich 库以实现极其 premium 的终端界面
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.status import Status
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

def welcome_banner():
    title = "🤖 HBPU-AI Personal Tax & Finance Agent"
    subtitle = "基于 ReAct 推理环、本地 RAG 个税政策库与反射工具链的多功能智能体系统"
    if HAS_RICH:
        panel = Panel(
            Text(f"{title}\n{subtitle}", justify="center", style="bold cyan"),
            border_style="cyan",
            expand=False
        )
        console.print(panel)
        console.print("💡 [bold green]输入你的税收/财务咨询指令进行对话：[/bold green]")
        console.print("💡 [bold yellow]特殊指令: /exit (退出), /clear (清空记忆), /tools (查看已注册工具)[/bold yellow]\n")
    else:
        print("=" * 60)
        print(title)
        print(subtitle)
        print("=" * 60)
        print("💡 输入你的税收/财务咨询指令进行对话：")
        print("💡 特殊指令: /exit (退出), /clear (清空记忆), /tools (查看已注册工具)\n")

def run_interactive_cli():
    welcome_banner()
    
    # 实例化智能体，设置个税顾问的 System Prompt
    system_prompt = (
        "你是一个极其专业、耐心的个人所得税与理财规划顾问。\n"
        "你能够利用各种内置工具，协助用户查询国家税务规定、获取其个人年度财务账单，并且进行精确的个人所得税与理财方案计算。\n"
        "【工作行为准则】\n"
        "1. 如果用户需要获取其个人的收入或财务明细，请主动调用对应的工具获取，不要胡乱猜测或胡说八道。\n"
        "2. 涉及税法政策规定（如专项扣除标准、捐赠额扣除上限、税率等级等），请**务必调用政策检索工具**获取权威信息，严禁幻觉。\n"
        "3. 计算个人应缴纳所得税或进行复杂金额折算时，请**务必调用计算器工具和个税计算工具**确保算术结果100%精确，禁止口算。"
    )
    
    try:
        agent = Agent(system_prompt=system_prompt, max_turns=8, verbose=True)
    except Exception as e:
        print(f"❌ 启动智能体失败: {e}")
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
            agent.clear_memory()
            if HAS_RICH:
                console.print("[bold green]✔ 对话历史已清空。[/bold green]")
            else:
                print("✔ 对话历史已清空。")
            continue
        elif cmd == "/tools":
            from tool_registry import tool_registry
            schemas = tool_registry.get_openai_tool_schemas()
            if HAS_RICH:
                console.print("\n[bold cyan]🔧 当前已注册的工具：[/bold cyan]")
                for schema in schemas:
                    func = schema["function"]
                    console.print(f"  • [bold yellow]{func['name']}[/bold yellow]: {func['description']}")
            else:
                print("\n🔧 当前已注册的工具：")
                for schema in schemas:
                    func = schema["function"]
                    print(f"  • {func['name']}: {func['description']}")
            continue

        # 智能体响应调用
        try:
            if HAS_RICH:
                # 使用 rich.Live/Status 营造高科技感十足的加载状态
                with Status("[cyan]Agent 正在思考并执行...", console=console) as status:
                    response_text = agent.run(user_input)
                console.print(Panel(Markdown(response_text), title="🤖 Agent Reply", border_style="green"))
            else:
                print("\nAgent 思考中...")
                response_text = agent.run(user_input)
                print("\n🤖 Agent Reply:")
                print(response_text)
                print("-" * 50)
        except Exception as e:
            if HAS_RICH:
                console.print(f"[bold red]❌ 交互运行过程中发生错误: {e}[/bold red]")
            else:
                print(f"❌ 交互运行过程中发生错误: {e}")

if __name__ == "__main__":
    run_interactive_cli()
