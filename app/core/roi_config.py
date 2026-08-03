from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RoiBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class RoiConfig:
    width: int
    height: int
    rois: dict[str, RoiBox]
    version: str = "1280x720-v4"


def load_roi_config(path: Path) -> RoiConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    size = data["standard_size"]
    raw_rois = data["rois"]
    rois = {
        name: RoiBox(
            x1=int(item["x1"]),
            y1=int(item["y1"]),
            x2=int(item["x2"]),
            y2=int(item["y2"]),
        )
        for name, item in raw_rois.items()
    }
    return RoiConfig(
        width=int(size["width"]),
        height=int(size["height"]),
        rois=rois,
        version=str(data.get("layout_version", "1280x720-v4")),
    )


def load_map_names(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("map_names", [])
    return [str(x) for x in names]


def load_map_aliases(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    aliases = data.get("map_aliases", {})
    return {str(alias): str(name) for alias, name in aliases.items()}
