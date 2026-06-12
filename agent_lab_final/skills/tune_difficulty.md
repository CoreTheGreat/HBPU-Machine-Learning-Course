# Skill: Numerical Balance & Tuning (数值难度调平能力)

## 📌 适用阶段
- **Milestone 3: 数值难度平衡与关卡配置 (RAG/JS)**

## 🛠️ 操作规程与指令指南

### 1. 难度与速度平滑爬升公式
- **禁止使用恒定不变的速度值**，否则游戏玩法将变得枯燥。
- 必须根据当前得分 (Score) 动态更新帧更新间隔间隔时间 (Interval)：
  ```javascript
  // 随着得分增加，间隔时间缩短（速度变快），但设置下限（如 50ms）防止闪烁崩溃
  let baseInterval = 150; 
  let currentInterval = Math.max(50, baseInterval - Math.floor(score / 5) * 10);
  ```

### 2. 关卡级别 (Level) 与得分倍率 (Multiplier)
- 每隔一定的分数（例如 10 分）进行 Level 跃升：`level = Math.floor(score / 10) + 1`。
- 高关卡级别（Level）应当带给玩家更高的额外得分倍率，提供正反馈刺激：
  `points_earned = base_points * level`。

### 3. RAG 知识检索引用规范
- 在编写速度、关卡或难度变化规则前，**必须**使用 `retrieve_game_design` 搜索 “网页小游戏数值平衡设计规范” 或 “难度衰减与难度公式”。
- 将检索到的公式原文及细节直接转化为 JavaScript 代码实现，严禁自行胡乱拼写参数。
