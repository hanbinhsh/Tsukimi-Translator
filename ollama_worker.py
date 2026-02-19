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

    def _ocr_options(self) -> dict:
        """统一 OCR 采样参数，尽量提高多次识别的一致性。"""
        return {
            "temperature": float(self.cfg.get("ocr_temperature", 0)),
            "seed": int(self.cfg.get("ocr_seed", 0)),
            "num_predict": int(self.cfg.get("ocr_num_predict", 4096)),
        }

    # ──────────────────────────────────────────
    # 独立方法（供 TranslationThread 分步调用）
    # ──────────────────────────────────────────

    def run_ocr(self, image_bytes: bytes) -> str:
        """调用视觉模型提取图片中的文字，返回纯文本"""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.cfg["ocr_model"],
            "prompt": self.cfg.get(
                "ocr_prompt",
                "Extract all text from this image. Output only the text content, no explanations."
            ),
            "images": [b64],
            "stream": False,
            "options": self._ocr_options(),
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
            "prompt": self.cfg.get("overlay_ocr_prompt", "<|grounding|>OCR the image."),
            "images": [b64],
            "stream": False,
            "options": self._ocr_options(),
        }
        res = requests.post(
            self._api_for("ocr"), json=payload, headers=self._headers_for("ocr"), timeout=30
        ).json()
        raw = res.get("response", "").strip()
        print(f"[overlay_ocr 原始模型输出全文]\n{raw}\n[overlay_ocr 原始模型输出结束]")
        items = self._parse_grounding_ocr(raw)
        if not items:
            print(f"[overlay_ocr] 原始模型输出:\n{raw}")
        return items

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

        def _to_int_bbox(coords):
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                return None
            vals = []
            for n in coords:
                try:
                    vals.append(int(round(float(n))))
                except Exception:
                    return None
            return vals

        def _try_parse_json_items(txt: str) -> list:
            candidates = [txt]
            if "```" in txt:
                candidates.extend(m.group(1).strip() for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", txt))

            for cand in candidates:
                cand = cand.strip()
                if not cand:
                    continue
                try:
                    data = json.loads(cand)
                except Exception:
                    continue

                if isinstance(data, dict):
                    data = data.get("items") or data.get("result") or data.get("data") or []
                if not isinstance(data, list):
                    continue

                parsed = []
                for it in data:
                    if not isinstance(it, dict):
                        continue
                    text = str(it.get("text") or it.get("content") or "").strip()
                    bbox = _to_int_bbox(it.get("bbox") or it.get("box") or it.get("det") or it.get("coords"))
                    if text and bbox:
                        parsed.append({"text": text, "bbox": bbox})
                if parsed:
                    return parsed
            return []

        results = []
        pattern = re.compile(
            r'<\|ref\|>(.*?)<\|/ref\|>\s*'
            r'<\|det\|>\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]<\|/det\|>'
            r'(?:\n([^\n<].*?))?(?=\n<\|ref\|>|$)',
            re.DOTALL
        )
        for m in pattern.finditer(raw):
            bbox = _to_int_bbox([m.group(2), m.group(3), m.group(4), m.group(5)])
            if not bbox:
                continue
            ref_text = (m.group(1) or '').strip()
            tail_text = (m.group(6) or '').strip()
            text = tail_text if tail_text else ref_text
            if text:
                results.append({"text": text, "bbox": bbox})

        if results:
            return results

        # 兜底：部分模型会直接输出 JSON 数组或 markdown 包裹的 JSON
        return _try_parse_json_items(raw)

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
