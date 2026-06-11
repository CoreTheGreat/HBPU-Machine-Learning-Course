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
            self.chunks = ["❌ 警告：未找到税收政策知识库文档。"]
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
        专为中文优化：将中文切分为单个字符，英文和数字保留为单词。
        无任何外部依赖，非常适合用于教学和演示。
        """
        def tokenize(text: str) -> set[str]:
            text = text.lower()
            # 提取所有中文字符
            zh_chars = re.findall(r"[\u4e00-\u9fa5]", text)
            # 提取英文字词和数字
            en_words = re.findall(r"[a-z0-9]+", text)
            return set(zh_chars + en_words)

        query_tokens = tokenize(query)
        if not query_tokens:
            return "\n\n---\n\n".join(self.chunks[:top_k])

        scored_chunks = []
        for chunk in self.chunks:
            chunk_tokens = tokenize(chunk)
            # 基础分：匹配的字/词的交集大小
            intersection = query_tokens.intersection(chunk_tokens)
            score = len(intersection)
            
            # 额外加权：如果查询短句中的连续中文字符串整体出现在 chunk 中，大幅加分
            # 比如查询“继续教育”，如果 chunk 中连续包含“继续教育”四字，相关度应该极高
            zh_phrases = re.findall(r"[\u4e00-\u9fa5]{2,}", query.lower())
            for phrase in zh_phrases:
                if phrase in chunk.lower():
                    score += len(phrase) * 5  # 词组匹配给予高权重
                    
            scored_chunks.append((score, chunk))

        # 按得分从高到低排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # 过滤掉得分为 0 的无关联块，取前 top_k 个
        relevant_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
        
        if not relevant_chunks:
            # 如果没有匹配到任何内容，返回起征点和首要规则
            return "\n\n---\n\n".join(self.chunks[:top_k])

        return "\n\n---\n\n".join(relevant_chunks)

# 默认加载 tax_policy
# 自动定位根目录下 data/tax_policy.txt
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tax_policy.txt")
tax_retriever = SimpleRetriever(data_path)

if __name__ == "__main__":
    print("--- 测试 SimpleRetriever (RAG) 模块 ---")
    print(f"加载数据文件路径: {tax_retriever.filepath}")
    print(f"数据总片段数量: {len(tax_retriever.chunks)} 个 Chunks")
    
    # 运行几个检索测试
    queries = ["继续教育专项扣除", "公益慈善捐赠百分之三十限额", "个税税率表"]
    for q in queries:
        print(f"\n🔍 检索问题: '{q}'")
        context = tax_retriever.retrieve(q, top_k=1)
        print("======== 检索到的最相关内容 ========")
        print(context)
        print("====================================")
