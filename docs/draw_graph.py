"""
draw_graph.py - Render the commander LangGraph flow as a styled Graphviz PNG.

This intentionally avoids LangGraph/Mermaid rendering because the generated
layout is harder to read for this flow. Keep EDGES in sync with
commander/orchestrator.py when graph routing changes.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "langgraph_flow.png"
CHAT_NODES = [
    "greeting_node",
    "human_reply_node",
    "task_classification_node",
    "ai_reply_node",
    "chat_memory_node",
    "goodbye_node",
]

PICK_NODES = [
    "input_node",
    "find_node",
    "get_item_info_no_sam3d_node",
    "nav_move_node",
    "observe_node",
    "reason_node",
    "update_item_info_1_node",
    "major_nav_node",
    "update_item_info_2_node",
    "car_grasp_node",
    "car_approach_node",
    "update_memory_node",
    "nav_home_node",
]

NODES = ["START", *CHAT_NODES, *PICK_NODES, "END"]

EDGES = [
    ("START", "greeting_node", "", ""),
    ("greeting_node", "human_reply_node", "", ""),
    ("human_reply_node", "task_classification_node", "continue", ""),
    ("human_reply_node", "goodbye_node", "goodbye", ""),
    ("task_classification_node", "ai_reply_node", "general_chat", ""),
    ("task_classification_node", "input_node", "specific_task", ""),
    ("ai_reply_node", "chat_memory_node", "", ""),
    ("chat_memory_node", "human_reply_node", "", ""),
    ("input_node", "find_node", "", ""),
    ("find_node", "get_item_info_no_sam3d_node", "selected", ""),
    ("find_node", "goodbye_node", "end", ""),
    ("get_item_info_no_sam3d_node", "nav_move_node", "ready", ""),
    ("get_item_info_no_sam3d_node", "nav_home_node", "failed/no_goal", ""),
    ("nav_move_node", "observe_node", "bootstrap", ""),
    ("nav_move_node", "update_memory_node", "nav failed", ""),
    ("update_memory_node", "observe_node", "", ""),
    ("observe_node", "reason_node", "", ""),
    ("reason_node", "update_item_info_1_node", "major_nav_node", ""),
    ("reason_node", "update_item_info_2_node", "car_grasp_node", ""),
    ("reason_node", "nav_home_node", "end", ""),
    ("update_item_info_1_node", "get_item_info_no_sam3d_node", "target_changed", ""),
    ("update_item_info_1_node", "major_nav_node", "unchanged", ""),
    ("update_item_info_1_node", "nav_home_node", "target_missing", ""),
    ("major_nav_node", "nav_move_node", "ready", ""),
    ("major_nav_node", "nav_home_node", "exhausted", ""),
    ("update_item_info_2_node", "get_item_info_no_sam3d_node", "target_changed", ""),
    ("update_item_info_2_node", "car_grasp_node", "unchanged", ""),
    ("update_item_info_2_node", "nav_home_node", "target_missing", ""),
    ("car_grasp_node", "car_approach_node", "", ""),
    ("car_approach_node", "nav_home_node", "end / success", "success"),
    ("car_approach_node", "update_memory_node", "failure", "failure"),
    ("nav_home_node", "goodbye_node", "", ""),
    ("goodbye_node", "END", "", ""),
]

NODE_STYLES = {
    "START": {"shape": "oval", "fillcolor": "#dcfce7", "color": "#16a34a", "label": "START", "penwidth": "1.8"},
    "END": {"shape": "oval", "fillcolor": "#fee2e2", "color": "#dc2626", "label": "END", "penwidth": "1.8"},
}

CLUSTERS = [
    ("cluster_chat", "chat", CHAT_NODES, "#e0f2fe", "#38bdf8"),
    ("cluster_pick", "pick", PICK_NODES, "#dcfce7", "#22c55e"),
]


def _quote(value: str) -> str:
    return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\""


def _attrs(values: dict[str, str]) -> str:
    return ", ".join(f"{key}={_quote(str(value))}" for key, value in values.items())


def _cluster_label(label: str) -> str:
    return f'<<B><FONT POINT-SIZE="18">{label}</FONT></B>>'


def _node_line(node: str, *, indent: str = "  ") -> str:
    attrs = {"label": node, **NODE_STYLES.get(node, {})}
    return f"{indent}{_quote(node)} [{_attrs(attrs)}];"


def build_dot() -> str:
    lines = [
        "digraph LangGraphFlow {",
        '  graph [rankdir=TB, bgcolor="white", pad="0.4", nodesep="0.5", ranksep="0.65", splines=ortho];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", penwidth="1.2", fontname="DejaVu Sans", fontsize=11, margin="0.12,0.08"];',
        '  edge [color="#475569", fontcolor="#475569", penwidth="1.2", fontname="DejaVu Sans", fontsize=9, arrowsize=0.7];',
        "",
    ]
    lines.append(_node_line("START"))
    lines.append(_node_line("END"))
    lines.append("")

    clustered = set()
    for cluster_name, label, nodes, fillcolor, color in CLUSTERS:
        clustered.update(nodes)
        cluster_attrs = {
            "style": "rounded,filled",
            "fillcolor": fillcolor,
            "color": color,
            "penwidth": "1.4",
            "fontname": "DejaVu Sans",
            "margin": "14",
            "labelloc": "t",
        }
        lines.extend([
            f"  subgraph {_quote(cluster_name)} {{",
            f"    graph [label={_cluster_label(label)}, {_attrs(cluster_attrs)}];",
        ])
        for node in nodes:
            lines.append(_node_line(node, indent="    "))
        lines.append("  }")
        lines.append("")

    for node in NODES:
        if node not in clustered and node not in {"START", "END"}:
            lines.append(_node_line(node))
    lines.append("")

    for src, dst, label, _style_key in EDGES:
        attrs = {"xlabel": label} if label else {}
        suffix = f" [{_attrs(attrs)}]" if attrs else ""
        lines.append(f"  {_quote(src)} -> {_quote(dst)}{suffix};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    dot = build_dot()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        print(dot)
        print("Graphviz 'dot' not found; PNG was not generated.")
        return 1

    subprocess.run([dot_bin, "-Tpng", "-o", str(OUTPUT_PATH)], input=dot.encode("utf-8"), check=True)
    print("PNG saved to:", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
