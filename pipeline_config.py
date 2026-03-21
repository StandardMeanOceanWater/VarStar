# pipeline_config.py — 共用 YAML 設定讀取
# 所有模組（photometry, period_analysis 等）統一從此處載入 observation_config.yaml
"""
用法：
    from pipeline_config import load_pipeline_config
    cfg_dict = load_pipeline_config()  # 自動搜尋 yaml
    cfg_dict = load_pipeline_config(Path("path/to/config.yaml"))
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("pipeline_config")


# ── 自動偵測 observation_config.yaml 路徑 ────────────────────────────────────
def _find_config() -> Path:
    """
    搜尋順序：
    1. 環境變數 VARSTAR_CONFIG
    2. 目前工作目錄
    3. 本檔案所在目錄（pipeline/）
    4. pipeline/ 的上層（project_root）
    5. Colab 的 Drive 預設路徑
    """
    env = os.environ.get("VARSTAR_CONFIG")
    if env:
        return Path(env)

    candidates = [
        Path.cwd() / "observation_config.yaml",
        Path(__file__).parent / "observation_config.yaml",
        Path(__file__).parent.parent / "observation_config.yaml",
        Path("/content/drive/Shareddrives/VarStar/pipeline/observation_config.yaml"),
        Path("D:/VarStar/pipeline/observation_config.yaml"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue

    raise FileNotFoundError(
        "找不到 observation_config.yaml。\n"
        "請設定環境變數 VARSTAR_CONFIG 或把 yaml 放在工作目錄下。\n"
        "例如：os.environ['VARSTAR_CONFIG'] = 'D:/VarStar/pipeline/observation_config.yaml'"
    )


def _detect_project_root(cfg_dict: dict, config_path: Path) -> Path:
    try:
        import google.colab  # noqa: F401
        root = cfg_dict["paths"]["colab"]["project_root"]
    except (ImportError, KeyError):
        root = cfg_dict["paths"]["local"]["project_root"]
    p = Path(root)
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    return p


def load_pipeline_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    讀取 observation_config.yaml 並回傳 dict。
    自動偵測路徑並計算 _project_root, _data_root, _config_path。
    """
    if config_path is None:
        config_path = _find_config()
    if not config_path.exists():
        raise FileNotFoundError(f"設定檔不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg_dict = yaml.safe_load(fh)

    if not cfg_dict:
        raise ValueError(f"設定檔為空：{config_path}")

    cfg_dict["_project_root"] = _detect_project_root(cfg_dict, config_path)
    cfg_dict["_data_root"] = cfg_dict["_project_root"] / "data"
    cfg_dict["_config_path"] = config_path

    # ── 基本結構驗證 ──────────────────────────────────────────────────────────
    _validate_sessions(cfg_dict)

    return cfg_dict


_SESSION_REQUIRED = {"date", "targets"}

def _validate_sessions(cfg_dict: Dict) -> None:
    """檢查 obs_sessions 的必填欄位，載入時即報錯。"""
    sessions = cfg_dict.get("obs_sessions", [])
    if not sessions:
        return
    for i, sess in enumerate(sessions):
        if not isinstance(sess, dict):
            logger.warning("obs_sessions[%d] 不是 dict，跳過", i)
            continue
        missing = _SESSION_REQUIRED - set(sess.keys())
        if missing:
            raise ValueError(
                f"obs_sessions[{i}] 缺少必填欄位：{missing}。"
                f"  請檢查 observation_config.yaml。"
            )


def get_nested(cfg: Dict, *keys, default=None):
    """安全地從巢狀 dict 取值。"""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is default:
            return default
    return node
