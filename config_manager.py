import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    # --- AI 模型 ---
    "ocr_model": "glm-ocr:latest",
    "llm_model": "qwen3-vl:2b-instruct",
    "ocr_api": "http://localhost:11434/api/generate",
    "ocr_key": "",
    "llm_api": "http://localhost:11434/api/generate",
    "llm_key": "",
    "ocr_prompt": "Extract all text from this image. Output only the text content, no explanations.",
    "overlay_ocr_prompt": "<|grounding|>OCR the image.",
    "ocr_temperature": 0,
    "ocr_seed": 0,
    "ocr_num_predict": 4096,
    "use_ocr": True,
    "use_llm": True,
    "use_stream": False,            # 是否流式输出
    "llm_prompt": "You are a translator. Please help me translate the following English text into Chinese. You should only tell me the translation result without any additional explanations.",

    # --- 截图 ---
    "scale_factor": 0.5,
    "target_hwnd": 0,
    "capture_region": None,         # 手动框选区域 {"x","y","w","h"}（屏幕相对物理像素），None 表示全窗口
    "capture_screen_name": "",      # 框选时使用的屏幕 QScreen.name()
    "capture_source": "window",        # "window" 窗口截图 / "region" 区域框选
    "capture_mode": "interval",
    "capture_interval": 2.5,
    "trigger_key": "Left Click",
    "auto_hide": True,

    # --- 窗口行为 ---
    "always_on_top": True,
    "window_visible": True,
    "ui_max_width": 800,
    "grow_direction": "up",         # "up" 向上扩展 / "down" 向下扩展

    # --- 字幕外观 ---
    "show_ocr_text": False,         # 是否同时显示 OCR 原文
    "ocr_color": "#FFFF88",         # OCR 原文颜色
    "trans_color": "#FFFFFF",       # 译文颜色
    "overlay_min_box_height": 28,    # 贴字文本框最小高度
    "overlay_auto_merge_lines": False, # 自动识别换行并拼接
    "overlay_min_line_height": 40,   # 小于该高度的行参与拼接
    "overlay_joiner": " ",           # 拼接字符（中日文可设为空）
    "remove_blank_lines": False,      # 自动移除翻译文本框中的空行
    "line_start_chars": ",.;:!?)]}、，。！？；：」』）】》",  # 下一行若以这些字符开头则判定为续句
    "line_end_chars": ".!?。！？…",  # 上一行若以这些字符结尾则判定为一句结束

    # --- 其他 ---
    "use_overlay_ocr": False,      # 贴字翻译（需要 deepseek-ocr 类模型）
    "show_overlay_debug_boxes": False,  # 贴字模式显示 OCR 原始检测框（调试）
    "auto_copy": False,
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            local_cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in local_cfg:
                    local_cfg[k] = v
            return local_cfg
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
