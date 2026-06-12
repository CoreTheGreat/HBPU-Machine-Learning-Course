import os
import sys
import re

# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class SimpleRetriever:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.chunks = []
        self._load_and_chunk()

    def _load_and_chunk(self):
        """将文本按段落分割为 Chunks"""
        if not os.path.exists(self.filepath):
            self.chunks = ["❌ 警告：未找到本地游戏开发知识库文档。"]
            return
            
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # 按照空行分割段落
            paragraphs = content.split("\n\n")
            self.chunks = [p.strip() for p in paragraphs if p.strip()]
        except Exception as e:
            self.chunks = [f"❌ 读取知识库失败: {e}"]

    def retrieve(self, query: str, top_k: int = 2) -> str:
        """
        基于字与词重合度的轻量级检索算法。
        """
        def tokenize(text: str) -> set[str]:
            text = text.lower()
            zh_chars = re.findall(r"[\u4e00-\u9fa5]", text)
            en_words = re.findall(r"[a-z0-9]+", text)
            return set(zh_chars + en_words)

        query_tokens = tokenize(query)
        if not query_tokens:
            return "\n\n---\n\n".join(self.chunks[:top_k])

        scored_chunks = []
        for chunk in self.chunks:
            chunk_tokens = tokenize(chunk)
            intersection = query_tokens.intersection(chunk_tokens)
            score = len(intersection)
            
            # 强化短语加权
            zh_phrases = re.findall(r"[\u4e00-\u9fa5]{2,}", query.lower())
            for phrase in zh_phrases:
                if phrase in chunk.lower():
                    score += len(phrase) * 5
                    
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
        
        if not relevant_chunks:
            # 默认返回前几个片段
            return "\n\n---\n\n".join(self.chunks[:top_k])

        return "\n\n---\n\n".join(relevant_chunks)

# 自动定位同级目录下的 RAG_DATA.txt
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG_DATA.txt")
game_retriever = SimpleRetriever(data_path)

if __name__ == "__main__":
    print("--- 测试 SimpleRetriever (RAG) 模块 ---")
    print(f"数据总片段数量: {len(game_retriever.chunks)} 个 Chunks")
    
    # 测试检索
    q = "贪吃蛇 键盘监听和移动"
    print(f"\n🔍 检索问题: '{q}'")
    context = game_retriever.retrieve(q, top_k=1)
    print("======== 检索到的内容 ========")
    print(context)
