import time

from ollama_worker import OllamaTranslator

DISPLAY_NAME = "Ollama glm-ocr 稳定检测"
SETTINGS_SCHEMA = [
    {
        "key": "detect_interval",
        "type": "float",
        "title": "检测间隔",
        "description": "每次 Ollama 返回后再等待多久进行下一次检测（秒）",
        "default": 1.0,
        "min": 0.0,
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
        self.wait_time = float(settings.get("stable_wait", 1.5) or 1.5)
        runtime_cfg = dict(cfg)
        runtime_cfg["ocr_model"] = "glm-ocr:latest"
        self.worker = OllamaTranslator(runtime_cfg)
        self.last_debug = {}

    def check(self, image_bytes: bytes):
        text, _ = self.worker.run_ocr(image_bytes, return_raw=True)
        now = time.time()
        normalized = (text or "").strip()

        if not normalized:
            self.last_text = ""
            self.last_changed_at = now
            self.last_debug = {"reason": "empty_text", "text_len": 0}
            return False, self.last_debug

        if normalized != self.last_text:
            self.last_text = normalized
            self.last_changed_at = now
            self.last_debug = {
                "reason": "text_changed",
                "text_len": len(normalized),
                "preview": normalized[:80],
            }
            return False, self.last_debug

        stable_elapsed = now - self.last_changed_at
        is_stable = stable_elapsed >= self.wait_time
        self.last_debug = {
            "reason": "same_text",
            "text_len": len(normalized),
            "stable_elapsed": round(stable_elapsed, 3),
            "threshold": self.wait_time,
            "preview": normalized[:80],
        }
        return is_stable, self.last_debug

    def is_stable(self, image_bytes: bytes) -> bool:
        ok, _ = self.check(image_bytes)
        return ok
