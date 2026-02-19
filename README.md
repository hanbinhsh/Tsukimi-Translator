# Tsukimi Translator

一个基于 **PySide6 + qfluentwidgets + Ollama** 的桌面截图翻译工具，支持：

- 窗口/区域截图翻译
- OCR + LLM 两阶段流水线
- 贴字（overlay）模式翻译
- 流式翻译输出
- 可视化配置（模型、提示词、上下文长度、贴字策略等）

## 当前版本

- `0.1`

## 主要功能

- **AI 配置**
  - OCR 模型、API、Key、提示词
  - LLM 模型、API、Key、提示词
  - OCR/LLM 上下文长度（`num_ctx`）
- **贴字配置**
  - 贴字 OCR 提示词
  - 自动换行拼接（最小行高、可拼接间距、拼接字符、断句规则）
- **关于页面**
  - 应用信息、仓库入口
  - 检查更新按钮（当前为本地提示）

## 运行方式

```bash
python main.py
```

> 依赖请先自行安装（例如 `PySide6`、`qfluentwidgets`、`pynput`、`requests` 等）。

## 配置文件

程序会在根目录读写 `config.json`，不存在时自动使用默认配置。

## 项目结构

- `main.py`：主界面、交互逻辑、翻译流程调度
- `ollama_worker.py`：OCR/LLM 请求与解析
- `config_manager.py`：默认配置、加载与保存
- `win_utils.py`：Windows 窗口相关辅助

## 仓库与作者

- 仓库：请在应用「关于」页面中配置为你的真实仓库地址
- 作者：Tsukimi Translator Contributors
