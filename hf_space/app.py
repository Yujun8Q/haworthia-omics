import os
from pathlib import Path

import gradio as gr
import pandas as pd

from download_segmentation_models import ensure_segmentation_models
from runtime_assets import prepare_private_assets


U2NET_HOME = Path(os.getenv("U2NET_HOME", "/tmp/haworthia-u2net")).expanduser()
os.environ["U2NET_HOME"] = str(U2NET_HOME)

STARTUP_ERROR = ""
SERVICE = None
ASSET_MANIFEST = {}
ASSET_DIRECTORY = Path("/tmp/haworthia-assets-unavailable")
try:
    ensure_segmentation_models(U2NET_HOME)
    from inference_core import HaworthiaInferenceService, SEGMENTATION_MODES

    ASSET_DIRECTORY, ASSET_MANIFEST = prepare_private_assets()
    SERVICE = HaworthiaInferenceService(ASSET_DIRECTORY)
except Exception as exc:
    STARTUP_ERROR = str(exc)
    SEGMENTATION_MODES = {
        "自动宽容": "adaptive",
        "双模型宽松": "lenient",
        "IS-Net 严格": "strict",
    }


def require_ready():
    if SERVICE is None:
        raise gr.Error("模型服务尚未就绪，请稍后重试。")


def require_consent(confirmed):
    if not confirmed:
        raise gr.Error("请先确认您有权上传并处理该图片。")


def predict(image, mode, sensitivity, threshold, confirmed):
    require_ready()
    require_consent(confirmed)
    try:
        return SERVICE.predict(image, mode, sensitivity, threshold)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def decode_attention(image, mode, sensitivity, confirmed):
    require_ready()
    require_consent(confirmed)
    try:
        return SERVICE.attention(image, mode, sensitivity)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def refresh_network(k):
    require_ready()
    return SERVICE.network_figure(k)


def refresh_neighbors(label, scope, count, k):
    require_ready()
    return SERVICE.nearest_table(label, scope, count, k)


def create_dendrogram():
    require_ready()
    return SERVICE.dendrogram_figure()


EMPTY_RESULTS = pd.DataFrame(columns=["排名", "类群", "余弦相似度", "子原型"])
EMPTY_NEIGHBORS = pd.DataFrame(
    columns=["排名", "相似类群", "关系", "余弦相似度", "互为近邻"]
)
APP_CSS = """
.app-shell { width: 100%; max-width: 1400px; margin: 0 auto; padding: 0 12px; }
.notice { border-left: 3px solid #6b7280; padding-left: 12px; color: #4b5563; }
.app-shell p { overflow-wrap: anywhere; }
"""

with gr.Blocks(
    title="Haworthia OMICS Demo",
    delete_cache=(3600, 3600),
) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.Markdown("# Haworthia OMICS")
        gr.Markdown(
            "面向瓦苇属植物表型研究的只读模型演示。结果表示当前模型中的表型相似性，"
            "不构成权威鉴定、遗传亲缘、杂交或演化关系结论。",
            elem_classes="notice",
        )
        if STARTUP_ERROR:
            gr.Markdown(
                "**服务初始化失败。** 管理员需要检查私有模型仓库、Space Secret 和运行日志。"
            )
        elif SERVICE is not None:
            gr.Markdown(
                f"已加载 {len(SERVICE.tax_ids)} 个类群的数值原型；模型权重未在公开 Space 中分发。"
            )

        with gr.Tabs():
            with gr.Tab("开放集推理"):
                with gr.Row():
                    predict_input = gr.Image(
                        type="pil", label="待分析图片", sources=["upload"]
                    )
                    segmented_output = gr.Image(
                        type="pil", label="模型实际接收的去背景图像"
                    )
                with gr.Row():
                    with gr.Column():
                        predict_mode = gr.Dropdown(
                            choices=list(SEGMENTATION_MODES),
                            value="自动宽容",
                            label="背景分割方式",
                        )
                        predict_sensitivity = gr.Slider(
                            0, 100, value=70, step=1, label="分割灵敏度"
                        )
                    with gr.Column():
                        rejection_threshold = gr.Slider(
                            -1.0, 1.0, value=0.55, step=0.01, label="未知类群拒绝阈值"
                        )
                predict_consent = gr.Checkbox(
                    label="我确认有权上传并处理该图片，且了解图片会经过临时计算。"
                )
                predict_button = gr.Button("提取特征并比对", variant="primary")
                prediction_status = gr.Markdown()
                prediction_table = gr.Dataframe(
                    value=EMPTY_RESULTS,
                    interactive=False,
                    label="最相似类群",
                )
                predict_button.click(
                    predict,
                    inputs=[
                        predict_input,
                        predict_mode,
                        predict_sensitivity,
                        rejection_threshold,
                        predict_consent,
                    ],
                    outputs=[segmented_output, prediction_status, prediction_table],
                    api_name=False,
                    concurrency_limit=1,
                )

            with gr.Tab("注意力热力图"):
                with gr.Row():
                    attention_input = gr.Image(
                        type="pil", label="待解析图片", sources=["upload"]
                    )
                    attention_segmented = gr.Image(
                        type="pil", label="去背景输入"
                    )
                with gr.Row():
                    with gr.Column():
                        attention_mode = gr.Dropdown(
                            choices=list(SEGMENTATION_MODES),
                            value="自动宽容",
                            label="背景分割方式",
                        )
                        attention_sensitivity = gr.Slider(
                            0, 100, value=70, step=1, label="分割灵敏度"
                        )
                attention_consent = gr.Checkbox(
                    label="我确认有权上传并处理该图片，且了解图片会经过临时计算。"
                )
                attention_button = gr.Button("生成注意力热力图", variant="primary")
                attention_output = gr.Image(
                    type="pil", label="四注意力头与门控权重"
                )
                attention_status = gr.Markdown()
                attention_button.click(
                    decode_attention,
                    inputs=[
                        attention_input,
                        attention_mode,
                        attention_sensitivity,
                        attention_consent,
                    ],
                    outputs=[attention_segmented, attention_output, attention_status],
                    api_name=False,
                    concurrency_limit=1,
                )

            with gr.Tab("表型网络与近邻"):
                if SERVICE is not None:
                    default_label = SERVICE.taxa[SERVICE.tax_ids[0]]["label"]
                    labels = [SERVICE.taxa[tax_id]["label"] for tax_id in SERVICE.tax_ids]
                    initial_network = SERVICE.network_figure(2)
                    initial_neighbors = SERVICE.nearest_table(
                        default_label, "全部", 8, 2
                    )
                else:
                    default_label = None
                    labels = []
                    initial_network = None
                    initial_neighbors = EMPTY_NEIGHBORS
                with gr.Row():
                    network_k = gr.Slider(
                        1, 5, value=2, step=1, label="每个节点保留的近邻边"
                    )
                    network_button = gr.Button("刷新网络")
                network_plot = gr.Plot(value=initial_network, label="表型相似度网络")
                gr.Markdown(
                    "红线表示跨物种表型相似；网络不证明基因流、亲本关系或演化方向。"
                )
                with gr.Row():
                    neighbor_taxon = gr.Dropdown(
                        choices=labels, value=default_label, label="选择类群"
                    )
                    neighbor_scope = gr.Radio(
                        ["全部", "仅跨物种", "仅同物种"],
                        value="全部",
                        label="关系范围",
                    )
                    neighbor_count = gr.Slider(
                        3, 12, value=8, step=1, label="显示数量"
                    )
                neighbor_table = gr.Dataframe(
                    value=initial_neighbors,
                    interactive=False,
                    label="近邻关系",
                )
                network_button.click(
                    refresh_network,
                    inputs=[network_k],
                    outputs=[network_plot],
                    api_name=False,
                ).then(
                    refresh_neighbors,
                    inputs=[neighbor_taxon, neighbor_scope, neighbor_count, network_k],
                    outputs=[neighbor_table],
                    api_name=False,
                )
                for control in (neighbor_taxon, neighbor_scope, neighbor_count, network_k):
                    control.change(
                        refresh_neighbors,
                        inputs=[neighbor_taxon, neighbor_scope, neighbor_count, network_k],
                        outputs=[neighbor_table],
                        api_name=False,
                    )

            with gr.Tab("树状聚类"):
                gr.Markdown(
                    "树状分支来自当前模型原型的余弦距离，不代表遗传系统发育。"
                )
                dendrogram_button = gr.Button("生成树状聚类")
                dendrogram_plot = gr.Plot(label="表型层次聚类树")
                dendrogram_button.click(
                    create_dendrogram,
                    outputs=[dendrogram_plot],
                    api_name=False,
                )

        gr.Markdown(
            """
---
本 Demo 仅用于学术研究、教育和个人学习。公开源代码适用 Apache-2.0；该说明不改变源码许可证。
应用不会将上传图片用于训练，也不会写入项目数据库；Hugging Face/Gradio 可能在平台层产生短期临时缓存。
未经授权的模型提取或再分发不受项目维护者认可。

[GitHub 源代码](https://github.com/YujunCC/haworthia-omics) ·
[问题与权利通知](https://github.com/YujunCC/haworthia-omics/issues)
"""
        )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
        css=APP_CSS,
        max_file_size="20mb",
        blocked_paths=[str(ASSET_DIRECTORY)],
        enable_monitoring=False,
        strict_cors=True,
        footer_links=[],
    )
