import io
import sqlite3
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from plotly.colors import qualitative
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
from torchvision.transforms import InterpolationMode

from model import TemperamentOmicsNet
from segmentation import segmentation_engine


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_IMAGE_PIXELS = 24_000_000
SEGMENTATION_MODES = {
    "自动宽容": "adaptive",
    "双模型宽松": "lenient",
    "IS-Net 严格": "strict",
}


def _normalize(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _format_label(species, variant):
    variant = (variant or "").strip()
    return f"{species} - {variant}" if variant else species


def _black_composite(image):
    rgba = image.convert("RGBA")
    output = Image.new("RGB", rgba.size, (0, 0, 0))
    output.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    return output


def _pad_to_square(image, fill):
    side = max(image.size)
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def _prepare_tensor(segmented):
    rgba = segmented.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = Image.new("RGB", rgba.size, (0, 0, 0))
    rgb.paste(rgba.convert("RGB"), mask=alpha)
    rgb = _pad_to_square(rgb, (0, 0, 0))
    alpha = _pad_to_square(alpha, 0)
    rgb = TF.resize(
        rgb, [224, 224], interpolation=InterpolationMode.BILINEAR
    )
    alpha = TF.resize(
        alpha, [224, 224], interpolation=InterpolationMode.BILINEAR
    )
    image_tensor = TF.normalize(TF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD)
    mask_tensor = TF.to_tensor(alpha).clamp(0.0, 1.0)
    return image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), rgb


def _validate_input(image):
    if image is None:
        raise ValueError("请先上传一张图片。")
    image = image.convert("RGB")
    if image.width < 32 or image.height < 32:
        raise ValueError("图片尺寸过小。")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("图片像素过多，请缩小至 2400 万像素以内。")
    image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    return image


class HaworthiaInferenceService:
    def __init__(self, asset_directory):
        self.asset_directory = Path(asset_directory)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        self.lock = threading.RLock()
        self.model = TemperamentOmicsNet().to(self.device)
        state = torch.load(
            self.asset_directory / "model_base.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.taxa, self.mean_prototypes, self.cluster_prototypes = (
            self._load_catalog(self.asset_directory / "catalog.db")
        )
        self.tax_ids = sorted(self.taxa)
        if len(self.tax_ids) < 3:
            raise RuntimeError("Private prototype catalog contains fewer than three taxa.")
        self._network_cache = {}

    @staticmethod
    def _load_catalog(path):
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            image_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            if image_count != 0:
                raise RuntimeError("Hosted catalog must not contain image records.")
            taxa = {
                int(tax_id): {
                    "species": species,
                    "variant": variant,
                    "label": _format_label(species, variant),
                }
                for tax_id, species, variant in connection.execute(
                    "SELECT id, species, variant FROM taxonomy ORDER BY id"
                )
            }
            means = {
                int(tax_id): _normalize(
                    np.frombuffer(blob, dtype=np.float32).copy()
                )
                for tax_id, blob in connection.execute(
                    "SELECT tax_id, feature_blob FROM prototypes ORDER BY tax_id"
                )
            }
            clusters = defaultdict(list)
            for tax_id, cluster_index, blob, sample_count in connection.execute(
                "SELECT tax_id, cluster_index, feature_blob, sample_count "
                "FROM prototype_clusters ORDER BY tax_id, cluster_index"
            ):
                clusters[int(tax_id)].append({
                    "index": int(cluster_index),
                    "feature": _normalize(np.frombuffer(blob, dtype=np.float32).copy()),
                    "sample_count": int(sample_count),
                })
            valid_ids = set(taxa) & set(means) & set(clusters)
            return (
                {key: taxa[key] for key in valid_ids},
                {key: means[key] for key in valid_ids},
                {key: clusters[key] for key in valid_ids},
            )
        finally:
            connection.close()

    def segment(self, image, mode_label, sensitivity):
        image = _validate_input(image)
        mode = SEGMENTATION_MODES.get(mode_label)
        if mode is None:
            raise ValueError("未知分割模式。")
        segmented, metrics = segmentation_engine.segment(
            image, mode=mode, sensitivity=int(sensitivity)
        )
        return segmented, metrics

    def predict(self, image, mode_label, sensitivity, rejection_threshold):
        with self.lock, torch.inference_mode():
            segmented, metrics = self.segment(image, mode_label, sensitivity)
            inputs, foreground, _ = _prepare_tensor(segmented)
            embedding = self.model(
                inputs.to(self.device), foreground.to(self.device)
            )[0].cpu().numpy()

        candidates = []
        for tax_id in self.tax_ids:
            best = max(
                (
                    float(embedding @ row["feature"]),
                    row["index"] + 1,
                )
                for row in self.cluster_prototypes[tax_id]
            )
            candidates.append({
                "taxon": self.taxa[tax_id]["label"],
                "similarity": best[0],
                "prototype": best[1],
            })
        candidates.sort(key=lambda row: row["similarity"], reverse=True)
        top_similarity = candidates[0]["similarity"]
        rejected = top_similarity < float(rejection_threshold)
        status = (
            f"未知类群：最高相似度 {top_similarity:.4f} 低于拒绝阈值。"
            if rejected
            else f"已通过开放集拒绝阈值，最高相似度 {top_similarity:.4f}。"
        )
        if metrics.get("suspicious"):
            status += " 当前分割被质量规则标记为可疑，请尝试其他分割模式。"
        table = pd.DataFrame([
            {
                "排名": index,
                "类群": row["taxon"],
                "余弦相似度": round(row["similarity"], 4),
                "子原型": row["prototype"],
            }
            for index, row in enumerate(candidates[:5], start=1)
        ])
        return _black_composite(segmented), status, table

    def attention(self, image, mode_label, sensitivity):
        with self.lock, torch.inference_mode():
            segmented, metrics = self.segment(image, mode_label, sensitivity)
            inputs, foreground, display = _prepare_tensor(segmented)
            _, masks, gates = self.model(
                inputs.to(self.device),
                foreground.to(self.device),
                return_masks=True,
            )
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )[0].cpu().numpy()
        gate_values = [float(gate.item()) for gate in gates]
        grayscale = np.asarray(display, dtype=np.float32).mean(axis=-1)
        figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
        for index, axis in enumerate(axes):
            axis.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
            axis.imshow(masks[index], cmap="turbo", alpha=0.45)
            axis.set_title(f"Head {index + 1} · gate {gate_values[index]:.3f}", fontsize=9)
            axis.axis("off")
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        buffer.seek(0)
        heatmap = Image.open(buffer).convert("RGB").copy()
        message = "热力图仅用于观察模型注意力，不是植物学性状标注。"
        if metrics.get("suspicious"):
            message += " 当前分割质量可疑，解释时应谨慎。"
        return _black_composite(segmented), heatmap, message

    def _network_data(self, k):
        k = min(max(int(k), 1), 5)
        if k in self._network_cache:
            return self._network_cache[k]
        features = np.stack([
            _normalize(np.mean(
                [row["feature"] for row in self.cluster_prototypes[tax_id]], axis=0
            ))
            for tax_id in self.tax_ids
        ])
        similarity = np.clip(features @ features.T, -1.0, 1.0)
        np.fill_diagonal(similarity, -1.0)
        distance = np.clip(1.0 - np.clip(features @ features.T, -1.0, 1.0), 0.0, 2.0)
        coordinates = MDS(
            n_components=2,
            metric="precomputed",
            metric_mds=True,
            random_state=42,
            n_init=1,
            init="random",
            max_iter=300,
        ).fit_transform(distance)
        pairs = set()
        neighbor_sets = []
        for index in range(len(self.tax_ids)):
            nearest = [int(value) for value in np.argsort(-similarity[index])[:k]]
            neighbor_sets.append(set(nearest))
            for neighbor in nearest:
                pairs.add(tuple(sorted((index, neighbor))))
        data = {
            "similarity": similarity,
            "coordinates": coordinates,
            "pairs": sorted(pairs),
            "neighbor_sets": neighbor_sets,
        }
        self._network_cache[k] = data
        return data

    def network_figure(self, k):
        data = self._network_data(k)
        coordinates = data["coordinates"]
        species_names = sorted({self.taxa[tax_id]["species"] for tax_id in self.tax_ids})
        palette = qualitative.Dark24 + qualitative.Alphabet
        color_map = {
            species: palette[index % len(palette)]
            for index, species in enumerate(species_names)
        }
        figure = go.Figure()
        for left, right in data["pairs"]:
            left_id, right_id = self.tax_ids[left], self.tax_ids[right]
            cross_species = (
                self.taxa[left_id]["species"] != self.taxa[right_id]["species"]
            )
            figure.add_trace(go.Scatter(
                x=[coordinates[left, 0], coordinates[right, 0]],
                y=[coordinates[left, 1], coordinates[right, 1]],
                mode="lines",
                line={
                    "color": "#d45745" if cross_species else "#a7afb8",
                    "width": 1.4 if cross_species else 0.8,
                },
                opacity=0.70 if cross_species else 0.35,
                hoverinfo="skip",
                showlegend=False,
            ))
        for species in species_names:
            indices = [
                index for index, tax_id in enumerate(self.tax_ids)
                if self.taxa[tax_id]["species"] == species
            ]
            figure.add_trace(go.Scatter(
                x=[coordinates[index, 0] for index in indices],
                y=[coordinates[index, 1] for index in indices],
                mode="markers",
                name=species.replace("Haworthia ", ""),
                text=[self.taxa[self.tax_ids[index]]["label"] for index in indices],
                customdata=[
                    sum(row["sample_count"] for row in self.cluster_prototypes[self.tax_ids[index]])
                    for index in indices
                ],
                hovertemplate="%{text}<br>原型样本计数 %{customdata}<extra></extra>",
                marker={
                    "color": color_map[species],
                    "size": [
                        8 + min(10, np.sqrt(max(1, value)))
                        for value in [
                            sum(row["sample_count"] for row in self.cluster_prototypes[self.tax_ids[index]])
                            for index in indices
                        ]
                    ],
                    "line": {"color": "white", "width": 0.8},
                },
            ))
        figure.update_layout(
            height=720,
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            title=f"表型相似度网络 · 每节点 {int(k)} 条近邻边",
            xaxis={"visible": False},
            yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
            legend={"title": "物种", "font": {"size": 10}},
            plot_bgcolor="#fbfcfd",
            paper_bgcolor="white",
        )
        return figure

    def nearest_table(self, label, scope, count, k):
        tax_id = next(
            (value for value in self.tax_ids if self.taxa[value]["label"] == label),
            None,
        )
        if tax_id is None:
            return pd.DataFrame()
        data = self._network_data(k)
        source_index = self.tax_ids.index(tax_id)
        rows = []
        for target_index in np.argsort(-data["similarity"][source_index]):
            target_index = int(target_index)
            target_id = self.tax_ids[target_index]
            same_species = (
                self.taxa[tax_id]["species"] == self.taxa[target_id]["species"]
            )
            if scope == "仅跨物种" and same_species:
                continue
            if scope == "仅同物种" and not same_species:
                continue
            rows.append({
                "排名": len(rows) + 1,
                "相似类群": self.taxa[target_id]["label"],
                "关系": "同物种" if same_species else "跨物种",
                "余弦相似度": round(float(data["similarity"][source_index, target_index]), 4),
                "互为近邻": (
                    target_index in data["neighbor_sets"][source_index]
                    and source_index in data["neighbor_sets"][target_index]
                ),
            })
            if len(rows) >= int(count):
                break
        return pd.DataFrame(rows)

    @lru_cache(maxsize=1)
    def dendrogram_figure(self):
        labels = [self.taxa[tax_id]["label"] for tax_id in self.tax_ids]
        features = np.stack([self.mean_prototypes[tax_id] for tax_id in self.tax_ids])
        distance = np.clip(1.0 - features @ features.T, 0.0, 2.0)
        condensed = distance[np.triu_indices(len(distance), k=1)]
        result = dendrogram(
            linkage(condensed, method="average"), labels=labels, no_plot=True
        )
        figure = go.Figure()
        for x_values, y_values in zip(result["dcoord"], result["icoord"]):
            figure.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": "#59636e", "width": 1.1},
                hoverinfo="skip",
                showlegend=False,
            ))
        tick_values = [5 + 10 * index for index in range(len(result["ivl"]))]
        figure.update_layout(
            height=max(900, len(labels) * 18),
            margin={"l": 250, "r": 20, "t": 50, "b": 40},
            title="当前模型原型的平均连接层次聚类",
            xaxis_title="余弦距离",
            yaxis={"tickmode": "array", "tickvals": tick_values, "ticktext": result["ivl"]},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return figure
