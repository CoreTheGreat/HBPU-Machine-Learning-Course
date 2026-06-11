import sys
# 解决 Windows 终端下打印 Emoji/Unicode 时的编码报错问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from tool_registry import tool, tool_registry
from rag import tax_retriever

# ==========================================
# 🛠️ 1. 定义智能体的默认可用工具
# ==========================================

@tool
def calculate(expression: str) -> str:
    """
    一个高精度数学表达式计算器。支持加减乘除、括号等运算。
    
    :param expression: 需要计算的数学表达式字符串，如 '200000 - 5000 - 60000'
    """
    try:
        # 为了学生实验的绝对安全性，使用安全受限的 eval 估值
        # 仅允许数字和基本运算符
        clean_expr = expression.replace(" ", "")
        if not all(c in "0123456789+-*/()." for c in clean_expr):
            return "错误: 表达式包含非法字符。只能包含数字和 +-*/()."
        result = eval(clean_expr, {"__builtins__": None}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错: {str(e)}"

@tool
def get_user_financials(user_id: str) -> str:
    """
    查询用户个人的年度财务收支与基本信息（包括年度总工资收入、专项附加扣除申报额、公益捐赠额和子女数量等数据）。
    
    :param user_id: 用户唯一识别码（目前仅支持 'S2026'）
    """
    # 模拟静态数据库档案
    financial_db = {
        "S2026": {
            "name": "张三",
            "annual_income": 200000.0,
            "continuing_education_cost": 5000.0,
            "continuing_education_type": "专业技术人员职业资格继续教育（计算机软考证书）",
            "charity_donation": 2000.0,
            "children_count": 1
        }
    }
    
    info = financial_db.get(user_id)
    if not info:
        return f"未找到用户编号为 {user_id} 的财务与档案信息。"
        
    return (
        f"【财务数据档案 - 用户 {user_id} ({info['name']})】\n"
        f"- 年度工资总收入: {info['annual_income']} 元\n"
        f"- 专项支出项目: 继续教育费用 {info['continuing_education_cost']} 元，类型为: {info['continuing_education_type']}\n"
        f"- 公益慈善捐赠: {info['charity_donation']} 元\n"
        f"- 子女数量: {info['children_count']} 个"
    )

@tool
def query_tax_policy(query: str) -> str:
    """
    在国家个人所得税政策知识库（包含起征点、专项附加扣除标准、个税税率表、捐赠扣除限制等）中检索相关规定。
    
    :param query: 检索的问题或关键词，例如 '继续教育扣除标准', '个税税率表', '捐赠限额'
    """
    return tax_retriever.retrieve(query)

@tool
def calculate_tax(taxable_income: float, tax_rate: float, quick_deduction: float) -> float:
    """
    根据给定的应纳税所得额、适用税率 and 速算扣除数，快速计算最终应缴纳的个人所得税。
    计算公式：个税 = 应纳税所得额 * 税率 - 速算扣除数
    
    :param taxable_income: 全年应纳税所得额（已扣除免税额、专项扣除等）
    :param tax_rate: 适用税率（如 0.03 代表 3%, 0.10 代表 10%）
    :param quick_deduction: 速算扣除数（元）
    """
    try:
        tax = taxable_income * tax_rate - quick_deduction
        return round(max(0.0, tax), 2)
    except Exception as e:
        return f"计算出错: {e}"


if __name__ == "__main__":
    print("--- 测试 tools 模块（内置业务工具）---")
    
    print("\n1. 测试 calculate:")
    print(calculate("125 * 8"))
    
    print("\n2. 测试 get_user_financials:")
    print(get_user_financials("S2026"))
    
    print("\n3. 测试 query_tax_policy:")
    print(query_tax_policy("继续教育扣除标准"))
    
    print("\n4. 测试 calculate_tax:")
    print(calculate_tax(10000.0, 0.03, 0.0))
