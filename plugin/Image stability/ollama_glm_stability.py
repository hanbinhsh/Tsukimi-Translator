import time

from ollama_worker import OllamaTranslator

DISPLAY_NAME = "Ollama glm-ocr 稳定检测"
SETTINGS_SCHEMA = [
    {
        "key": "detect_interval",
        "type": "float",
        "title": "检测间隔",
        "description": "多久向 Ollama 发送一次检测（秒）",
        "default": 1.0,
        "min": 0.2,
        "max": 10.0,
        "step": 0.1,
    },
    {
        "key": "stable_wait",
        "type": "float",
        "title": "稳定等待",
        "description": "文本连续不变且非空达到该时长后触发（秒）",
        "default": 1.5,
        "min": 0.2,
        "max": 30.0,
        "step": 0.1,
    },
]


class StabilityChecker:
    """文本非空且连续不变达到稳定等待时间时返回 True。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.last_text = ""
        self.last_changed_at = time.time()
        settings = cfg.get("stability_settings", {}) or {}
        self.interval = float(settings.get("detect_interval", 1.0) or 1.0)
        self.wait_time = float(settings.get("stable_wait", 1.5) or 1.5)
        runtime_cfg = dict(cfg)
        runtime_cfg["ocr_model"] = "glm-ocr:latest"
        self.worker = OllamaTranslator(runtime_cfg)

    def is_stable(self, image_bytes: bytes) -> bool:
        text, _ = self.worker.run_ocr(image_bytes, return_raw=True)
        now = time.time()
        normalized = (text or "").strip()
        if not normalized:
            self.last_text = ""
            self.last_changed_at = now
            time.sleep(self.interval)
            return False

        if normalized != self.last_text:
            self.last_text = normalized
            self.last_changed_at = now
            time.sleep(self.interval)
            return False

        time.sleep(self.interval)
        return (now - self.last_changed_at) >= self.wait_time
