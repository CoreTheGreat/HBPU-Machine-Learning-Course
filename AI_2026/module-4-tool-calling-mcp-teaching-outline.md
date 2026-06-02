# Module 4: 工具调用 — 教学课件提纲

> **课程**: 人工智能理论  
> **模块**: 模块4 — 大语言模型的工具调用  
> **时长**: 1.5 课时 (70分钟)  
> **前置**: 模块3 (记忆与检索增强生成)  
> **承接**: 模块5 (智能体)  
> **设计日期**: 2026-05-18

---

## 教学大纲要求

| 教学内容 | 要求 | 重难点 | 学时 |
|---------|------|--------|------|
| 4.1 工具调用 | 理解 Function Calling 原理，能实现简单工具调用 | 系统提示词，工具调用原理 | 2 |
| 4.2 MCP | 掌握 MCP 协议，能运用 MCP 扩展 LLM 功能 | MCP 与工具调用的关系 | 2 |

---

## 教学逻辑主线

```
模块3结尾：有了记忆，LLM还需要"手"。
    │
    ├── 4.1 工具调用 (Function Calling) — 基础机制
    │   ├── 问题：LLM 只能"说"，不能"做"
    │   ├── 原理：JSON Schema → LLM 输出函数调用 → 执行 → 返回
    │   ├── 判据：工具描述（Prompt）决定 LLM 何时调哪个工具
    │   ├── 工作流模式：单工具 / 多工具 / 顺序 / 并行
    │   └── 真实案例：Hermes 的 terminal 工具 — CLI 也是 Function Calling
    │
    ├── 4.2 MCP — 工具的标准化接口协议
    │   ├── 问题：每个应用都要重写一遍工具连接逻辑
    │   ├── MCP 的答案：标准化的工具接口 → 工具实现一次，到处用
    │   └── MCP 与 Function Calling 的关系
    │
    ├── 4.3 CLI — Function Calling 框架下最通用的工具形态
    │   ├── CLI 不是独立机制——它的调用判据和 get_weather 完全一样
    │   ├── 为什么 CLI 在 Agent 中被广泛使用：零封装成本 + LLM 天然精通 Shell
    │   └── 能力边界：CLI 强大但"工具发现"和"跨平台一致性"是短板
    │
    └── 过渡：工具调用 → Agent 的"手"
        └── → 模块5：智能体 (Agent)
```

**三者关系**：

```
                    Function Calling
               "LLM 输出结构化工具调用请求"
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     具体 API 工具        MCP 工具        Terminal 工具
   (手写函数注册)     (标准协议发现)    (Shell 命令执行)
          │                │                │
    {"name":"get_weather"│}   │                │
    {"arguments":{...}}  │   │                │
          │                │                │
     调用判据相同：      调用判据相同：    调用判据相同：
     工具的 JSON Schema  工具的 JSON Schema  工具的 JSON Schema
     + description 文本   + description 文本   + description 文本
```

- **Function Calling**：LLM 输出 `{"function": "xxx", "arguments": {...}}` — 所有工具调用的**基础机制**
- **MCP**：定义了工具如何被**发现、描述、复用**的标准协议 — 解决的是"工具怎么连接"的问题
- **CLI (Terminal)**：它是 FC 框架下的一个**具体工具**——参数是 Shell 命令字符串。其调用判据和其他 FC 工具一样：靠 JSON Schema + description 描述

**关键认知**：无论 `get_weather` 还是 `terminal`，LLM 的决策逻辑完全相同——"用户说了什么 → 有没有匹配的工具描述 → 调哪个"。CLI 不是独立于 FC 之外的机制。

---

## 4.1 工具调用 (Function Calling) — 基础机制 (~30 min)

### Slide 1: 导入——LLM 的"手"在哪里？

> 🌰 场景：你问 ChatGPT "今天黄石的天气怎么样？"
> 它说："抱歉，我无法获取实时天气信息。"
>
> 但是——如果它能调用一个 `get_weather(city)` 函数呢？

**核心问题**：LLM 擅长"说"，但它不会"做"。它不能查天气、不能发邮件、不能操作数据库。

**核心答案**：让 LLM 不只是输出自然语言，而是输出**结构化的工具调用请求**。这个机制叫 Function Calling。

### Slide 2: 工具调用的基本流程

```
User: "北京今天天气怎么样？"
    │
    ▼
LLM: → {"function": "get_weather", "arguments": {"city": "北京"}}
    │
    ▼
Application: 调用天气 API → {"temp": 22, "condition": "晴"}
    │
    ▼
LLM: "北京今天晴，气温22°C。"
```

**关键认知**：LLM 不执行函数。它只做一件事——**决定调用哪个工具、传什么参数**。实际执行由应用程序（或 Agent 框架）完成。

### Slide 3: 工具调用的原理

**核心机制**：在 System Prompt 或 API 请求中描述可用的工具（JSON Schema 格式）：

```json
{
  "name": "get_weather",
  "description": "获取指定城市当前天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名称"}
    },
    "required": ["city"]
  }
}
```

**LLM 要做的两件事**：
1. **判断**：用户的问题是否需要调用工具？
2. **生成**：如果需要，输出正确的函数调用 JSON

**重难点**：工具描述的质量直接决定调用准确率——描述太模糊 LLM 不会调用，太具体会误调用。

### Slide 4: LLM 如何判断"什么时候调哪个工具"？

**答案：全靠工具描述（Prompt）——和判断"用户问天气→调 get_weather"完全一样。**

LLM 做决策时看的是这几样：
1. 用户说了什么（"帮我找一下文件"）
2. 每个工具的 `description` + 参数的 `description`（"Execute shell commands..."）
3. System Prompt 中是否有 Few-shot 示例

**不存在独立于 FC 的"CLI 调用机制"。terminal 工具的判据就是它的 JSON Schema。**

### Slide 5: 工具调用的工作流模式

| 模式 | 说明 | 例子 |
|------|------|------|
| **单工具调用** | LLM 选一个工具，执行，返回结果 | 查天气 |
| **多工具选择** | 多个工具可用，LLM 选择最合适的 | 查天气 OR 查股票 |
| **顺序调用** | 先调 A，用 A 的结果作为 B 的参数 | 先查城市代码 → 再查天气 |
| **并行调用** | 同时调多个互相独立的工具 | 同时查北京和上海 |

### Slide 6: 工具学习 (Tool Learning)

**问题**：LLM 怎么知道什么时候该用哪个工具？

**两个层面**：
1. **工具描述**（System Prompt）：用自然语言 + JSON Schema 精确描述每个工具的功能范围
2. **Few-shot 示例**：在 System Prompt 中给出"用户问题 → 工具调用"的示例，引导 LLM 行为

**常见陷阱**：
- 工具描述太模糊 → LLM 从不调用
- 工具描述太具体 → LLM 在不该调的时候也调
- 工具之间功能重叠 → LLM 选错

**教学提示**：可以现场演示——同一个问题，配不同的工具描述，LLM 的调用行为完全不同。

### Slide 7: 真实案例——CLI 工具的 Schema 和 get_weather 一样

这是 Hermes Agent 中 `terminal` 工具的 **实际注册代码片段**（来自 `tools/terminal_tool.py`）：

```python
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment.
Filesystem usually persists between calls.

Do NOT use cat/head/tail to read files — use read_file instead.
Do NOT use grep/rg/find to search — use search_files instead.
Do NOT use sed/awk to edit files — use patch instead.
Reserve terminal for: builds, installs, git, processes, scripts, network,
package managers, and anything that needs a shell.

Foreground (default): Commands return INSTANTLY when done.
Background: Set background=true to get a session_id...  """

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string",
                        "description": "The command to execute on the VM"},
            "background": {"type": "boolean", ...},
            "timeout": {"type": "integer", ...},
            "workdir": {"type": "string", ...},
        },
        "required": ["command"]
    }
}
```

**对比 `get_weather` 和 `terminal`**：

| | get_weather | terminal |
|---|:--:|:--:|
| **参数** | `{"city": "string"}` | `{"command": "string"}` |
| **描述内容** | "获取指定城市天气" | "执行 Shell 命令。用于安装、构建、Git……不要用于读文件" |
| **LLM 何时调用** | 用户问天气相关信息 | 用户需要安装/运行/操作文件系统等 Shell 能做的事 |
| **判断机制** | JSON Schema + description | JSON Schema + description |

**结论**：CLI 和任何其他 Function Calling 工具一样——它的调用判据就是工具描述。没有魔法。

### Slide 8: 短期记忆中的工具调用

> **回顾 Module 3**：工具调用的结果（API 返回值、文件内容、命令输出）会进入 Context Window，占用短期记忆空间。

**这就引出了一个实践问题**：
- 一次工具调用返回 5000 行日志 → 把上下文撑爆
- 需要在 System Prompt（Module 3 的上下文工程）中管理工具输出的 token 预算

---

## 4.2 MCP — 工具的标准化接口协议 (~20 min)

### Slide 9: Function Calling 的碎片化问题

> 🌰 场景：
> - 你用 OpenAI API 写了一个天气查询助手——用 OpenAI 的 Function Calling 格式
> - 现在你想把它迁到 DeepSeek——格式不一样，要重写
> - 你想给它加上文件系统访问——又要自己写一套文件操作逻辑
> - 你的同事用 Claude Desktop 写了另一个助手——但你的工具他没法复用

**核心问题**：Function Calling 只定义了"LLM 怎么表达'我要调工具'"，但没有定义**工具本身怎么被连接、发现、复用**。每个应用都在重复造轮子。

### Slide 10: MCP——AI 工具的"标准接口"

**定义**：Model Context Protocol (MCP) 是一个开放标准（由 Anthropic 提出），定义了 LLM 应用与外部工具之间的统一接口。

```
任何 LLM 应用（Claude Desktop、Hermes Agent、你的App……）
    │
    ├── MCP Client（内置在应用中）
    │       │
    │       ├── MCP Server A  →  文件系统
    │       ├── MCP Server B  →  数据库
    │       └── MCP Server C  →  Web 搜索
```

**核心价值**：工具只需实现一次（MCP Server）→ 所有支持 MCP 的 LLM 应用都能用。

### Slide 11: MCP 与 Function Calling 的关系

| | Function Calling | MCP |
|---|:--:|:--:|
| **是什么** | LLM 输出工具调用的格式约定 | 工具端的标准接口协议 |
| **解决什么问题** | "LLM 怎么表达'我要调工具'" | "工具怎么被连接、发现、复用" |
| **工具定义** | 硬编码在应用代码里 | 工具自己声明（`tools/list`） |
| **跨应用复用** | 不支持（每个应用重写） | 支持（一处实现，到处使用） |

**关系**：MCP 是 Function Calling 的**标准化基础设施**。MCP Server 暴露的工具，最终通过 Function Calling 被 LLM 调用。MCP 不改变 FC 的原理，而是让工具定义和通信统一化。

### Slide 12: 案例——MCP 实际解决了什么

**不用 MCP**：
```
Agent A：自己写文件操作 → os.listdir, open, read……
Agent B：自己写文件操作 → os.listdir, open, read……
Agent C：自己写数据库查询 → psycopg2, sqlite3……
→ 三套实现，三种 bug，三份维护
```

**用 MCP**：
```
MCP Server A（File System）：实现一次 → A、B 都用它
MCP Server B（PostgreSQL） ：实现一次 → C 用它
→ 一套实现，一处修复，全部受益
```

---

## 4.3 CLI — Function Calling 框架下最通用的工具形态 (~10 min)

### Slide 13: CLI 是什么——它就是一个参数是命令字符串的 FC 工具

回顾 Slide 7 的 Hermes `terminal` 工具 Schema：

```json
{
  "name": "terminal",
  "description": "Execute shell commands on a Linux environment...",
  "parameters": {
    "command": {"type": "string", "description": "The command to execute"}
  }
}
```

这和 `get_weather` 没有任何区别——只是参数从 `{"city": "北京"}` 变成了 `{"command": "find . -name '*.py' -mtime -7"}`。

**本质上，CLI 工具就是把整个操作系统的能力装进了一个通用工具**：LLM 理解用户意图后，翻译为 Shell 命令，通过 FC 调用 `terminal` 执行。

### Slide 14: 为什么 CLI 在 Agent 中被广泛使用

**可直接验证的来源**：Hermes 的 `terminal` 工具是 built-in 核心工具（非 MCP 插件），在所有平台默认启用。

```
$ hermes tools list
Built-in toolsets (cli):
  ✓ enabled  terminal  💻 Terminal & Processes
  ✓ enabled  file      📁 File Operations
  ...
```

**广泛使用的原因**：
- **零封装成本**：`curl`、`git`、`python`、`grep`……所有 CL工具天然可用，不需要为每个外部能力写 API 封装
- **LLM 天然精通 Shell 语法**：预训练数据中海量 Shell 命令 → LLM 知道怎么写
- **最小公约数**：所有能力最终都有一条 CLI 入口——安装（`pip`）、运行（`python`）、版本管理（`git`）、网络（`curl`）
- **调试方便**：命令可以复制到终端手动复现

### Slide 15: CLI 的能力展示与边界

| 你想让 LLM 做的事 | CLI 实现 | 本质 |
|:--|:--|:--|
| 操作文件 | `cat`, `find`, `grep`, `ls` | 调 `terminal` → 执行命令 |
| 运行代码 | `python script.py` | 同上 |
| 查 Git 历史 | `git log --oneline` | 同上 |
| 调用外部 API | `curl -H "..." https://api.example.com` | 同上 |
| 安装依赖 | `pip install xxx` | 同上 |

**CLI 能覆盖的**：所有有 CLI 入口的操作。
**CLI 覆盖不到的**：需要结构化类型安全（纯文本流）、需要动态工具发现、需要跨平台一致性的场景——这些正是 MCP 的设计目标。

---

## 4.4 总结与过渡 (~10 min)

### Slide 16: 工具调用技术栈全景

```
Function Calling      ← LLM 输出"我要调哪个工具、什么参数"
    │                    （所有工具调用的基础机制，判据 = 工具描述）
    │
    ├── 工具形态1：具体 API 工具（get_weather、send_email……）
    │
    ├── 工具形态2：MCP 工具（标准协议接入，一处实现到处复用）
    │
    └── 工具形态3：Terminal / CLI（参数是 Shell 命令，把操作系统能力装进一个工具）
         ↑
         调用的判据是一样的：JSON Schema + description 文本
```

### Slide 17: Module 2-4 能力拼图 → 预告 Module 5

```
Module 2: LLM 有"脑"（推理）
Module 3: LLM 有"记忆"（短期+长期记忆管理）
Module 4: LLM 有"手"（工具调用 = FC + MCP + CLI 三种工具形态）
         ↓
Module 5: Agent = 脑 + 记忆 + 手 + 自主规划
    "有了脑、记忆和手，LLM 还需要什么？——它需要知道自己该做什么。"
```

### Slide 18: 课后作业

1. **实践1 (Function Calling)**：用任意 LLM API 实现一个简单的工具调用——让 LLM 调用计算器函数（加减乘除），理解 FC 的基本流程。
2. **实践2 (CLI)**：利用终端工具，让 LLM 完成一个文件操作任务（如"统计当前目录下所有 Python 文件的总行数"），观察 LLM 如何将自然语言翻译为 Shell 命令。
3. **思考**：CLI 工具的参数是一个字符串命令——这和 `get_weather(city="北京")` 有什么本质区别？为什么它可以覆盖那么多不同类型的操作？

---

## 教学资源清单

### 引用 Wiki 中的页面
- [[tool-calling]] — Function Calling 机制
- [[model-context-protocol]] — MCP 标准
- [[context-engineering]] — 工具输出如何管理上下文（衔接 Module 3）

### 引用 Hermes 源代码
- `tools/terminal_tool.py` — `TERMINAL_TOOL_DESCRIPTION` 和 `TERMINAL_SCHEMA`（Slide 7 真实案例）

### 引用去年课件
- C11 §2.3-2.4: 工具调用和 MCP 部分（约 10 页幻灯片）

### 外部参考
- （待补充）
