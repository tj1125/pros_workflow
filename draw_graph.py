"""
draw_graph.py — 使用 LangGraph 內建功能將目前的 Graph 輸出成 PNG 圖片
"""
import os
from commander import load_project_env

load_project_env()

from commander.logger import TraceLogger
from commander.orchestrator import Orchestrator

# 用 mock 模式建立 orchestrator，只需要拿到 graph 物件
trace_logger = TraceLogger()
orchestrator = Orchestrator(trace_logger=trace_logger, use_mock=True)

# 取得 LangGraph 的可繪圖物件
graph = orchestrator.graph.get_graph()

# 輸出 Mermaid 語法（印在 terminal）
print("=== Mermaid Diagram ===")
print(graph.draw_mermaid())

# 輸出 PNG（需要有 pygraphviz 或使用 mermaid 線上服務）
try:
    png_bytes = graph.draw_mermaid_png()
    output_path = "docs/langgraph_flow.png"
    os.makedirs("docs", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"\nPNG saved to: {output_path}")
except Exception as e:
    print(f"\nPNG generation failed (expected if no graphviz): {e}")
    print("You can paste the Mermaid text above into https://mermaid.live to view the chart.")

