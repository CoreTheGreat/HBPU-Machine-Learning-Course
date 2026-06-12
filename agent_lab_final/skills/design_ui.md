# Skill: Design Web UI (网页 UI 与布局开发能力)

## 📌 适用阶段
- **Milestone 1: 页面框架与 UI 设计 (HTML/CSS)**

## 🛠️ 操作规程与指令指南

### 1. 结构骨架设计
- 游戏主页面必须命名为 `index.html`。
- 使用 HTML5 标准语义化标签，如 `<header>`, `<main>`, `<footer>`。
- 必须包含一个独立的游戏包装容器：`<div id="game-container">`。
- 画布元素使用 `<canvas id="gameCanvas">`，必须在 HTML 中显式声明 `width` 和 `height` 属性（如 `width="400" height="400"`），防止 CSS 缩放导致像素失真。

### 2. 赛博朋克 / 暗色霓虹配色规范
- 背景：使用深色渐变底，例如 `radial-gradient(circle, #1a1525 0%, #0c0813 100%)`。
- 主题边框色：发光霓虹蓝 (`#00f0ff`)，霓虹粉/红 (`#ff007f`)，发光荧光绿 (`#39ff14`)。
- 文字样式：字体首选等宽字体，如 `'Courier New', Courier, monospace`。
- 发光阴影效果（CSS 关键技巧）：
  ```css
  box-shadow: 0 0 15px #ff007f, inset 0 0 10px rgba(255, 0, 127, 0.3);
  text-shadow: 0 0 5px #39ff14;
  ```

### 3. HUD 信息面板
- 顶部或侧边必须提供 HUD 计分面板：显示当前分数 (Score)、关卡级别 (Level) 等。
- HUD 内容与画布应当左右对齐或水平居中对齐，看起来极为规整。

### 4. 交互式控制按钮
- 必须包含一个显眼的「开始游戏/重新开始」按钮，按钮应当拥有平滑的 CSS Hover 缩放动画及发光边界效果。
