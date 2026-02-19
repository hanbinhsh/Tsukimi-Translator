import requests
import base64
import json


class OllamaTranslator:
    def __init__(self, config):
        self.cfg = config

    def _api_for(self, service: str) -> str:
        default = "http://localhost:11434/api/generate"
        if service == "ocr":
            return self.cfg.get("ocr_api", default).strip() or default
        return self.cfg.get("llm_api", default).strip() or default

    def _headers_for(self, service: str) -> dict:
        key = (self.cfg.get("ocr_key", "") if service == "ocr"
               else self.cfg.get("llm_key", "")).strip()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # ──────────────────────────────────────────
    # 独立方法（供 TranslationThread 分步调用）
    # ──────────────────────────────────────────

    def run_ocr(self, image_bytes: bytes) -> str:
        """调用视觉模型提取图片中的文字，返回纯文本"""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.cfg["ocr_model"],
            "prompt": "Extract all text from this image. Output only the text content, no explanations.",
            "images": [b64],
            "stream": False,
        }
        res = requests.post(
            self._api_for("ocr"), json=payload, headers=self._headers_for("ocr"), timeout=20
        ).json()
        return res.get("response", "").strip()

    def run_llm(self, text: str, image_bytes: bytes = None) -> str:
        """调用 LLM 进行翻译/润色（非流式），返回结果字符串"""
        payload = self._build_llm_payload(text, image_bytes, stream=False)
        res = requests.post(
            self._api_for("llm"), json=payload, headers=self._headers_for("llm"), timeout=30
        ).json()
        return res.get("response", "").strip()

    def run_llm_stream(self, text: str, image_bytes: bytes = None):
        """调用 LLM 进行翻译/润色（流式），逐 token yield 字符串片段"""
        payload = self._build_llm_payload(text, image_bytes, stream=True)
        with requests.post(
            self._api_for("llm"), json=payload, headers=self._headers_for("llm"), timeout=60, stream=True
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break

    # ──────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────

    def _build_llm_payload(self, text: str, image_bytes: bytes, stream: bool) -> dict:
        prompt = self.cfg.get("llm_prompt", "")
        full_prompt = f"{prompt}\n\nContent: {text}" if text.strip() else prompt
        payload = {
            "model": self.cfg["llm_model"],
            "prompt": full_prompt,
            "stream": stream,
        }
        # 若没有文字（OCR 关闭），直接把图片发给多模态 LLM
        if not text.strip() and image_bytes:
            payload["images"] = [base64.b64encode(image_bytes).decode()]
        return payload

    # ──────────────────────────────────────────
    # 向后兼容的一体化方法
    # ──────────────────────────────────────────

    def process(self, image_bytes: bytes) -> str:
        extracted = ""
        if self.cfg.get("use_ocr", True):
            try:
                extracted = self.run_ocr(image_bytes)
            except Exception as e:
                return f"OCR Error: {e}"

        if self.cfg.get("use_llm", True):
            try:
                img = None if extracted else image_bytes
                return self.run_llm(extracted, img)
            except Exception as e:
                return f"LLM Error: {e}"

        return extracted

    # ──────────────────────────────────────────
    # 贴字模式：带坐标的 OCR
    # ──────────────────────────────────────────

    def run_ocr_with_coords(self, image_bytes: bytes) -> list:
        """
        调用 deepseek-ocr 类模型，返回带坐标的文本块列表。
        返回格式：[{"text": str, "bbox": [x1, y1, x2, y2]}, ...]
        坐标为 0-1000 归一化坐标，相对于图片尺寸。
        若模型不支持该格式，返回空列表（调用方应降级到普通 OCR）。
        """
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.cfg["ocr_model"],
            "prompt": "<|grounding|>OCR the image.",
            "images": [b64],
            "stream": False,
        }
        res = requests.post(
            self._api_for("ocr"), json=payload, headers=self._headers_for("ocr"), timeout=30
        ).json()
        raw = res.get("response", "").strip()
        return self._parse_grounding_ocr(raw)

    @staticmethod
    def _parse_grounding_ocr(raw: str) -> list:
        """
        解析 deepseek-ocr grounding 格式：
            <|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

        兼容两种输出：
          1) 文本在 <|ref|> 标签内（deepseek-ocr 常见）
          2) 文本在 det 行后续普通文本中（部分变体）
        返回：[{"text": str, "bbox": [x1, y1, x2, y2]}, ...]
        """
        import re

        results = []
        pattern = re.compile(
            r'<\|ref\|>(.*?)<\|/ref\|>\s*'
            r'<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>'
            r'(?:\n([^\n<].*?))?(?=\n<\|ref\|>|$)',
            re.DOTALL
        )
        for m in pattern.finditer(raw):
            x1, y1, x2, y2 = (int(m.group(2)), int(m.group(3)),
                              int(m.group(4)), int(m.group(5)))
            ref_text = (m.group(1) or '').strip()
            tail_text = (m.group(6) or '').strip()
            text = ref_text if ref_text else tail_text
            if text:
                results.append({"text": text, "bbox": [x1, y1, x2, y2]})
        return results

    def run_llm_batch(self, items: list) -> list:
        """
        对列表中每条 text 依次调用 LLM 翻译，返回填充了 translated 字段的同列表。
        items: [{"text": str, "bbox": [...], ...}, ...]
        """
        for item in items:
            try:
                item["translated"] = self.run_llm(item["text"])
            except Exception as e:
                item["translated"] = item["text"]   # 翻译失败保留原文
                print(f"[LLM batch error] {e}")
        return items