import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os
import subprocess
import time
from tool_registry import tool
from rag import game_retriever

# 动态计算当前 workspace 路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(CURRENT_DIR, "workspace")

# 确保工作区目录存在
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# 全局变量，记录服务器进程，防止重复启动
_server_process = None


@tool
def write_game_file(filename: str, code_content: str) -> str:
    """
    向游戏工作区写入一个代码或资源文件（例如 index.html, styles.css, game.js）。
    
    :param filename: 文件名（不得包含路径分隔符，只写文件名）
    :param code_content: 文件中完整的代码或 HTML 内容
    :return: 写入成功或失败的状态信息
    """
    # 限制路径，防止目录穿越
    clean_name = os.path.basename(filename)
    filepath = os.path.join(WORKSPACE_DIR, clean_name)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code_content)
        return f"✅ 成功将代码写入工作区文件: {clean_name}，大小: {len(code_content)} 字节。"
    except Exception as e:
        return f"❌ 写入文件失败: {e}"


@tool
def read_game_file(filename: str) -> str:
    """
    读取并返回游戏工作区中某个文件（如 index.html, game.js）的完整内容。
    
    :param filename: 文件名（只写文件名）
    :return: 文件的完整内容或错误信息
    """
    clean_name = os.path.basename(filename)
    filepath = os.path.join(WORKSPACE_DIR, clean_name)
    
    if not os.path.exists(filepath):
        return f"❌ 未找到文件: {clean_name}。"
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"❌ 读取文件失败: {e}"


@tool
def list_workspace() -> str:
    """
    列出当前游戏工作区 (workspace/) 中已生成的所有文件列表及其基本信息。
    
    :return: 文件列表字符串
    """
    if not os.path.exists(WORKSPACE_DIR):
        return "📂 工作区尚未创建。"
        
    try:
        files = os.listdir(WORKSPACE_DIR)
        if not files:
            return "📂 当前工作区为空，暂未开发任何文件。"
            
        lines = []
        for name in files:
            path = os.path.join(WORKSPACE_DIR, name)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))
                lines.append(f"  • {name} ({size} 字节) | 修改时间: {mtime}")
        return "📂 工作区当前文件列表:\n" + "\n".join(lines)
    except Exception as e:
        return f"❌ 列出工作区文件失败: {e}"


@tool
def retrieve_game_design(query: str) -> str:
    """
    在本地游戏开发知识库中搜索关于数值设计、Canvas 绘图规范、按键监听以及物理碰撞检测的相关内容。
    
    :param query: 搜索的关键词或描述
    :return: 相关的文档片段
    """
    return game_retriever.retrieve(query, top_k=2)


@tool
def start_game_server() -> str:
    """
    启动本地轻量级 Web 服务器以托管当前工作区中的网页游戏，并返回访问 URL。
    
    :return: 服务器启动状态及游玩链接
    """
    global _server_process
    
    # 检测是否已经运行
    if _server_process and _server_process.poll() is None:
        return "🎮 游戏服务器已在运行中！请访问：http://localhost:8000/index.html 游玩。"
        
    try:
        # 启动 Python 内置的 http.server
        # 使用 subprocess 在后台运行，并将根目录指向工作区
        cmd = [sys.executable, "-m", "http.server", "8000", "--directory", WORKSPACE_DIR]
        
        # Windows 系统下需要特殊的 creationflags 避免弹出黑窗口
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
            
        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags
        )
        
        # 稍等片刻让服务器绑定端口
        time.sleep(0.5)
        return "🎮 本地小游戏 HTTP 服务器启动成功！\n👉 请在浏览器中点击链接游玩： http://localhost:8000/index.html"
    except Exception as e:
        return f"❌ 启动本地 HTTP 服务器失败: {e}。你可以手动双击工作区下的 index.html 文件直接游玩。"


@tool
def list_skills() -> str:
    """
    列出当前可用的全部开发技能（skills/）文件名，以便在开发的不同阶段调用特定的技能指南。
    
    :return: 技能列表描述
    """
    skills_dir = os.path.join(CURRENT_DIR, "skills")
    if not os.path.exists(skills_dir):
        return "⚠️ 未发现 skills 目录。"
        
    try:
        files = os.listdir(skills_dir)
        markdown_files = [f for f in files if f.endswith(".md")]
        if not markdown_files:
            return "📂 技能库中暂无可用技能。"
        return "💡 可用技能列表:\n" + "\n".join([f"  • {f[:-3]}" for f in markdown_files])
    except Exception as e:
        return f"❌ 获取技能列表失败: {e}"


@tool
def get_skill(skill_name: str) -> str:
    """
    获取某个特定开发技能的详细规范、操作指南和代码要求，以指导当前开发工作。
    
    :param skill_name: 技能名称（例如 'design_ui', 'develop_gameplay', 'tune_difficulty', 'verify_code'）
    :return: 该技能的具体指令规范内容
    """
    clean_name = os.path.basename(skill_name)
    if not clean_name.endswith(".md"):
        clean_name += ".md"
        
    filepath = os.path.join(CURRENT_DIR, "skills", clean_name)
    if not os.path.exists(filepath):
        return f"❌ 未找到对应技能: {skill_name}。你可以调用 list_skills 查询可用技能。"
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"❌ 读取技能详情失败: {e}"


if __name__ == "__main__":
    print("--- 测试 Tools 模块 ---")
    print(list_workspace())
    print("\n测试写入:")
    print(write_game_file("test.txt", "Hello Game Dev!"))
    print(list_workspace())
    print("\n测试读取:")
    print(read_game_file("test.txt"))
    # 清理测试文件
    try:
        os.remove(os.path.join(WORKSPACE_DIR, "test.txt"))
    except:
        pass
