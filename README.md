# Tsukimi Translator

一个基于 **PySide6 + qfluentwidgets + Ollama** 的桌面截图翻译工具，适用于游戏、生肉漫画等场景。

它通过「截图 → OCR → 翻译 → 悬浮显示」流水线，将屏幕中的文本快速转成目标语言，并支持贴字（Overlay）模式、规则处理与流式输出。

耗时：

普通模式（glm-ocr + qwen3-vl:2b-instruct）（4k/3080ti-laptop-16G）: 2~4s

贴字模式（deepseek-ocr + qwen3-vl:2b-instruct）（4k/3080ti-laptop-16G）: 10~30s

解压即用 [点我下载](https://github.com/hanbinhsh/Tsukimi-Translator/releases) 解压后双击.exe即可（请确保你有API或者本地已经部署了Ollama以及对应模型）

<img width="1626" height="1626" alt="NZ8`X$CZ4S9`6HXOMBHEG$R" src="https://github.com/user-attachments/assets/6f1dfb41-c353-4a94-a255-0c9989e1c990" />

![1db262402dc88f359dba1b80e1c89c53](https://github.com/user-attachments/assets/bf97c851-279d-4b9c-98aa-6799130d3f25)

![c7a59069b9fc992d6b55d8be9419ba56](https://github.com/user-attachments/assets/a57aef8f-4adf-4863-bd36-7a435525061d)

<img width="1420" height="1435" alt="11dfaa347ffb793ab0983dcab6c875cd" src="https://github.com/user-attachments/assets/86052353-9441-4349-836d-d45ecc342b61" />

<img width="3840" height="2160" alt="0RVEK2LEAE J`DU47)71LF1" src="https://github.com/user-attachments/assets/789ae526-9941-4292-bd90-9c7d308b1772" />

---

## 免责声明

### 本软件（不使用外部API时）并不会将你的任何数据上传至互联网，由使用外部API造成如隐私泄露等一切后果和开发者无关
### 有一些在线游戏可能会启用反作弊系统，使用本软件导致触发反作弊系统导致的一切后果和开发者无关

---

## 目录

- [功能概览](#功能概览)
- [运行环境与依赖](#运行环境与依赖)
- [快速开始](#快速开始)
- [Ollama 配置（重点）](#ollama-配置重点)
- [应用内配置说明](#应用内配置说明)
- [项目结构](#项目结构)
- [常见问题（FAQ）](#常见问题faq)
- [作者](#作者)

---

## 功能概览

- **截图翻译**
  - 支持窗口截图与手动框选区域截图。
  - 支持触发延时与重复触发保护。
- **OCR + LLM 双阶段**
  - OCR 模型提取图片文本。
  - LLM 对文本进行翻译/润色，支持自定义提示词。
- **贴字（Overlay）模式**
  - 支持坐标 OCR（适配 grounding 风格输出）。
  - 支持自动换行拼接、最小行高与行距策略。
- **规则引擎**
  - OCR 后规则组与输出规则组，支持正则/大小写/整词匹配。
- **可视化配置**
  - 模型、API、Key、上下文长度、颜色、透明度等均可在 UI 配置。
- **调试能力（在关于页依次按下ICE后解锁）**
  - 支持 OCR 原始日志、文本日志、调试截图开关（适合排错）。

---

## 运行环境与依赖

### 推荐环境

- **系统**：Windows（项目中含 `win_utils.py`，主要围绕 Windows 窗口能力）
- **Python**：3.10+（建议 3.11）
- **Ollama**：最新稳定版

### 安装依赖

```bash
pip install -r requirements.txt
```

如果你尚未准备好虚拟环境，建议先执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 快速开始

1. 确保 Ollama 已安装并运行。
2. 拉取你要使用的 OCR / LLM 模型（见下一节）。
3. 启动应用：

```bash
python main.py
```

4. 在应用配置页中填写：
   - OCR 模型 / OCR API
   - LLM 模型 / LLM API
   - 提示词、上下文长度等
5. 选择截图来源（窗口或区域），设置触发方式后开始翻译。

---

## Ollama 配置（重点）

本项目默认通过 Ollama 的 `/api/generate` 接口调用模型，默认地址：

- `http://localhost:11434/api/generate`

### 1) 安装 Ollama

- 官网下载安装后，启动 Ollama 服务。
- 验证是否运行：

```bash
ollama list
```

若能正常返回本地模型列表，说明服务可用。

### 2) 拉取模型

可按你的用途准备两类模型：

- **OCR 模型**：负责“看图读字”（支持图片输入）
- **LLM 模型**：负责“翻译文本”

示例：

```bash
ollama pull glm-ocr:latest
ollama pull deepseek-ocr
ollama pull qwen3-vl:2b-instruct
```

> 说明：
> - 默认配置中 OCR 模型是 `glm-ocr:latest`。
> - 贴字模型建议选用 `deepseek-ocr`目前只适配了这一种模型。
> - 默认配置中 LLM 模型是 `qwen3-vl:2b-instruct`。
> - 若你只想纯文本翻译，可将 LLM 换成普通文本模型。

### 3) 应用中的 API 与模型填写建议

在应用设置中建议填写：

- OCR API：`http://localhost:11434/api/generate`
- LLM API：`http://localhost:11434/api/generate`
- OCR Model：例如 `glm-ocr:latest`
- LLM Model：例如 `qwen3-vl:2b-instruct`

如需对接远端模型网关，也可替换为你的自建 API 地址，并按需填写 Key（Bearer Token）。

### 4) 关于上下文长度（num_ctx）

应用支持分别配置：

- `ocr_context_length`
- `llm_context_length`

这两个值会被写入请求的 `options.num_ctx`。一般建议：

- 显存/内存紧张：先用 4096 或更小
- 长文本翻译：可尝试 8192 或更高（视模型支持）

### 5) 流式输出

启用流式后，LLM 翻译会逐 token 显示，更快看到首批结果。

- 优点：体感延迟低
- 代价：输出可能分段闪动，最终文本需等待完整返回

### 6) 贴字模式模型建议

贴字模式依赖模型输出带坐标信息（grounding 风格或可解析 JSON）。

如果你发现贴字框定位不正确：

1. 优先更换更稳定的 OCR 视觉模型；
2. 调整 `overlay_ocr_prompt`；
3. 打开调试日志查看模型原始返回；
4. 必要时先关闭贴字，使用普通 OCR + 翻译模式验证主链路。

---

## 应用内配置说明

应用会在根目录读写 `config.json`，首次运行会自动生成默认值。

### 关键配置项（高频）

- **模型与接口**
  - `ocr_model` / `llm_model`
  - `ocr_api` / `llm_api`
  - `ocr_key` / `llm_key`
- **提示词与上下文**
  - `ocr_prompt`
  - `overlay_ocr_prompt`
  - `llm_prompt`
  - `ocr_context_length`
  - `llm_context_length`
- **流程开关**
  - `use_ocr`
  - `use_llm`
  - `use_stream`
  - `use_overlay_ocr`
- **截图相关**
  - `capture_source`（`window` / `region`）
  - `capture_delay_seconds`
  - `trigger_key`
- **界面相关**
  - `always_on_top`
  - `overlay_opacity`
  - `show_ocr_text`
  - `ocr_color` / `trans_color`

### 规则组（文本清洗）

- `ocr_rule_groups`：作用在 OCR 结果上
- `output_rule_groups`：作用在最终输出上

每条规则支持：

- 正则匹配（`regex`）
- 区分大小写（`case_sensitive`）
- 整词匹配（`whole_word`）

可用于处理常见问题：

- OCR 误识别符号替换
- 合并多余空行
- 去掉特定前后缀
- 游戏人名处理

---

## 项目结构

- `main.py`：主窗口 UI、交互流程、截图与翻译调度
- `ollama_worker.py`：OCR/LLM 请求封装、流式解析、贴字坐标解析
- `config_manager.py`：默认配置、配置读写
- `win_utils.py`：Windows 窗口辅助能力
- `plugin/Image stability/`：稳定性算法插件目录（可扩展）

---

## 常见问题（FAQ）

### Q1：点击触发后没有结果怎么办？

按顺序检查：

1. Ollama 是否正在运行；
2. 模型名是否正确，且已 `pull`；
3. OCR/LLM API 地址是否能连通；
4. 是否误关了 `use_ocr` 与 `use_llm`；
5. 控制台日志里是否有超时或 JSON 解析错误。

### Q2：翻译质量不稳定怎么办？

- 优先优化 `llm_prompt`，明确目标语言与风格。
- 对 OCR 文本先加清洗规则，减少乱码再送入 LLM。
- 模型性能不足时换更高质量模型或降低输入噪声（截图更清晰）。

### Q3：贴字模式定位偏移怎么办？

- 检查模型是否真的返回坐标。
- 调整 `overlay_min_line_height`、`overlay_max_line_gap`。
- 关闭自动拼接进行对照测试。

---

## 作者

- 作者：IceRinne aka. hanbinhsh

- 若有任何问题或建议请提供Issue反馈
