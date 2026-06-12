# Skill: Develop Game Loop & Mechanics (核心玩法与渲染开发能力)

## 📌 适用阶段
- **Milestone 2: 核心玩法与渲染逻辑 (Canvas/JS)**

## 🛠️ 操作规程与指令指南

### 1. 核心游戏循环 (Game Loop)
- 禁止使用简陋的 `setInterval` 驱动帧动画，防止高频掉帧。
- 必须基于 `requestAnimationFrame(loop)` 设计平滑的高精度渲染更新驱动器。
- 循环函数必须划分成 `update(dt)` (物理解耦更新) 和 `draw()` (纯渲染绘制) 两大部分。

### 2. 状态机模型 (Game State)
- 游戏必须包含以下三种以上的状态管理：
  - `start` (未开始，显示欢迎页面)
  - `playing` (运行中)
  - `gameover` (游戏结束，显示最终分数和重新开始按钮)
- 键盘监听器只应在游戏处于 `playing` 状态时生效。

### 3. Canvas 2D 绘图技巧
- 每次 `draw()` 开始时，使用 `ctx.clearRect(0, 0, width, height)` 彻底清除上帧画布。
- 贪吃蛇游戏：使用方块网格逻辑，每节身体大小（如 20x20）与画布宽高应能整除。
- 2048 游戏：使用 4x4 的矩阵数据结构进行二维映射。

### 4. 碰撞与边界判定
- 每次移动后更新坐标，并与墙壁边界（0 至 width/height）对比。
- 碰撞发生时，立即修改游戏状态为 `gameover`，禁止角色穿墙（除非设计了穿墙玩法，这需要显式记录在备忘录中）。
