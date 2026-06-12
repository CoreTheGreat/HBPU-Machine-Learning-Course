# Skill: Verify Connection & Server Run (联调校验与部署发布能力)

## 📌 适用阶段
- **Milestone 4: 文件组装与打包连通性自测**

## 🛠️ 操作规程与指令指南

### 1. 连通性连接核验 (Connection Checks)
- 仔细对照 `index.html` 中的引用链接：
  - `<link rel="stylesheet" href="styles.css">` 中的文件名必须与生成的 CSS 文件名完全一致。
  - `<script src="game.js"></script>` 中的文件名和路径必须与生成的 JS 文件名完全一致。
  - 严禁使用不存在的第三方远程 CDN 文件或图片链接。

### 2. 检查控制台报错与空值 (Error Verification)
- 检查 Canvas 上下文 `getContext('2d')` 在元素加载前不会被执行（通常用 `window.onload` 或 `DOMContentLoaded` 包裹）。
- 确保 DOM 按钮的点击事件监听器在元素存在时才被绑定：
  ```javascript
  const btn = document.getElementById("startBtn");
  if (btn) {
      btn.addEventListener("click", startGame);
  }
  ```

### 3. 本地托管测试部署 (Local Server Deployment)
- 运行 `list_workspace` 确认生成文件的总大小和列表完整度。
- 调用 `start_game_server` 在后台快速启动本地测试 Web 服务器。
- 提取并反馈最终的 URL 链接给用户，确保用户能通过点击该链接直接进行交互和体验。
