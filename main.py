import sys
import time
from pathlib import Path
import requests
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QBuffer, QIODevice, QObject, QPoint, QRect, QUrl
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QLayout, QPushButton, QFrame,
                                QSizePolicy)
from PySide6.QtGui import (QGuiApplication, QPainter, QPen, QColor,
                           QFont, QPainterPath, QFontMetrics, QIcon, QDesktopServices, QPixmap)
from shiboken6 import isValid
from qfluentwidgets import (FluentWindow, SubtitleLabel, ComboBox, PushButton,
                             setTheme, Theme, CardWidget, LineEdit, TextEdit,
                             SettingCardGroup, ScrollArea, PrimaryPushButton, InfoBar,
                             SwitchButton, DoubleSpinBox, IconWidget, SegmentedWidget,
                             MessageBox, NavigationItemPosition, ColorDialog)
from qfluentwidgets import FluentIcon as FIF
from pynput import mouse, keyboard

from win_utils import get_active_windows, get_window_rect
from ollama_worker import OllamaTranslator
from config_manager import load_config, save_config


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def get_seg_key(seg_widget, default=""):
    """从 SegmentedWidget 反查当前选中的 key"""
    current = seg_widget.currentItem()
    if hasattr(seg_widget, "items"):
        for k, v in seg_widget.items.items():
            if v == current:
                return k
    return default


def load_app_icon() -> QIcon:
    """优先加载项目中的 logo 图标（ico/svg），用于标题栏与任务栏。"""
    base = Path(__file__).resolve().parent
    candidates = [
        base / "logo.ico",
        base / "logo.svg",
        base / "assets" / "logo.ico",
        base / "assets" / "logo.svg",
        base / "icons" / "logo.ico",
        base / "icons" / "logo.svg",
    ]
    for p in candidates:
        if p.exists():
            icon = QIcon(str(p))
            if not icon.isNull():
                return icon
    return FIF.LANGUAGE.icon()


# ══════════════════════════════════════════════
# 信号桥 & 颜色按钮
# ══════════════════════════════════════════════

class InputSignal(QObject):
    triggered = Signal()


class ColorButton(QPushButton):
    """点击弹出颜色选择器，背景展示当前颜色"""
    color_changed = Signal(str)

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 32)
        self.setColor(color)
        self.clicked.connect(self._pick)

    def setColor(self, hex_color: str):
        self._color = hex_color
        self.setStyleSheet(
            f"QPushButton {{ background: {hex_color}; border: 2px solid #666;"
            f" border-radius: 6px; }}"
            f"QPushButton:hover {{ border-color: #bbb; }}"
        )

    def color(self) -> str:
        return self._color

    def _pick(self):
        dlg = ColorDialog(QColor(self._color), "选择颜色", self.window() or self)
        if dlg.exec():
            c = dlg.color
            if c.isValid():
                self.setColor(c.name())
                self.color_changed.emit(c.name())


class AvatarLabel(QLabel):
    """圆形头像标签，点击后跳转链接。"""

    def __init__(self, size=36, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self._url = ""

    def set_url(self, url: str):
        self._url = url

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._url:
            QDesktopServices.openUrl(QUrl(self._url))
        super().mousePressEvent(event)

    def set_avatar_from_bytes(self, data: bytes):
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        s = min(self.width(), self.height())
        pix = pix.scaled(s, s, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        rounded = QPixmap(s, s)
        rounded.fill(Qt.transparent)
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, s, s)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)
        painter.end()
        self.setPixmap(rounded)


# ══════════════════════════════════════════════
# 通用设置卡片
# ══════════════════════════════════════════════

class CustomSettingCard(CardWidget):
    def __init__(self, icon, title, subtitle, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(80)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 0, 16, 0)

        self.iconWidget = IconWidget(icon, self)
        self.iconWidget.setFixedSize(22, 22)
        self.layout.addWidget(self.iconWidget)
        self.layout.addSpacing(12)

        self.textLayout = QVBoxLayout()
        self.textLayout.setSpacing(2)
        self.titleLabel = QLabel(title, self)
        self.titleLabel.setStyleSheet("font: 14px 'Segoe UI Semibold'; color: white;")
        self.subLabel = QLabel(subtitle, self)
        self.subLabel.setStyleSheet("font: 12px 'Segoe UI'; color: #aaa;")
        self.textLayout.addStretch(1)
        self.textLayout.addWidget(self.titleLabel)
        self.textLayout.addWidget(self.subLabel)
        self.textLayout.addStretch(1)
        self.layout.addLayout(self.textLayout)
        self.layout.addStretch(1)

    def addWidget(self, widget):
        if isinstance(widget, (LineEdit, DoubleSpinBox, ComboBox, SegmentedWidget)):
            widget.setFixedWidth(220)
        self.layout.addWidget(widget, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addSpacing(16)


class PromptSettingCard(CardWidget):
    """标题/说明在上，TextEdit 在下的卡片。"""

    def __init__(self, icon, title, subtitle, height=110, parent=None):
        super().__init__(parent=parent)
        body_h = max(80, int(height))
        self.setMinimumHeight(body_h + 72)
        self.setMaximumHeight(body_h + 72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 12, 16, 12)
        self.layout.setSpacing(8)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)

        self.iconWidget = IconWidget(icon, self)
        self.iconWidget.setFixedSize(20, 20)
        top.addWidget(self.iconWidget, 0, Qt.AlignTop)

        txt = QVBoxLayout()
        txt.setSpacing(2)
        self.titleLabel = QLabel(title, self)
        self.titleLabel.setStyleSheet("font: 14px 'Segoe UI Semibold'; color: white;")
        self.subLabel = QLabel(subtitle, self)
        self.subLabel.setStyleSheet("font: 12px 'Segoe UI'; color: #aaa;")
        txt.addWidget(self.titleLabel)
        txt.addWidget(self.subLabel)
        top.addLayout(txt, 1)

        self.layout.addLayout(top)

        self.editor = TextEdit(self)
        self.editor.setFixedHeight(body_h)
        self.editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.editor)


# ══════════════════════════════════════════════
# 翻译线程（支持 OCR 中途信号 + 流式输出）
# ══════════════════════════════════════════════

class TranslationThread(QThread):
    ocr_ready     = Signal(str)        # OCR 提取完毕，发出原文
    partial_text  = Signal(str)        # 流式：累计译文
    finished      = Signal(str, float) # 最终译文, AI 总耗时
    overlay_ready = Signal(list)       # 贴字模式：[{"text","translated","bbox"}, ...]

    def __init__(self, image_bytes, config):
        super().__init__()
        self.image_bytes = image_bytes
        self.config = config

    @staticmethod
    def _looks_like_sentence_continuation(
        prev_text: str,
        next_text: str,
        line_start_chars: str,
        line_end_chars: str,
    ) -> bool:
        """判断下一行是否更像是上一行的续句（如英文断行）。"""
        prev = (prev_text or "").strip()
        nxt = (next_text or "").strip()
        if not prev or not nxt:
            return False

        # 上一行以连字符结束通常表示被截断（hy-
        # phen）
        if prev.endswith("-"):
            return True

        # 下一行若以小写/标点开头，通常是续句而不是新句
        head = nxt[0]
        if head.islower() or head in (line_start_chars or ""):
            return True

        # 上一行没有常见终止符时，更可能和下一行相连
        if line_end_chars:
            return prev[-1] not in line_end_chars
        return True

    @staticmethod
    def _merge_overlay_lines(
        items: list,
        min_height: int,
        max_line_gap: int,
        joiner: str,
        line_start_chars: str,
        line_end_chars: str,
    ) -> list:
        """将连续的小高度行拼接为一句，减少被错误断行的影响。"""
        if not items:
            return items

        merged = []
        current = None
        prev_small = False

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            h = y2 - y1
            is_small = h < max(1, min_height)

            if current is None:
                current = dict(item)
                prev_small = is_small
                continue

            cx1, cy1, cx2, cy2 = current["bbox"]
            vertical_gap = max(0, y1 - cy2)

            horizontal_overlap = max(0, min(cx2, x2) - max(cx1, x1))
            min_width = max(1, min(cx2 - cx1, x2 - x1))
            overlap_ratio = horizontal_overlap / min_width

            gap_limit = max(0, int(max_line_gap))
            near_line = vertical_gap <= gap_limit and overlap_ratio >= 0.2
            continuation = TranslationThread._looks_like_sentence_continuation(
                current.get("text", ""),
                item.get("text", ""),
                line_start_chars,
                line_end_chars,
            )

            should_merge = near_line and (
                (prev_small and is_small) or
                (prev_small and continuation) or
                (is_small and continuation) or
                (continuation and vertical_gap <= max(2, int(min_height * 0.5)))
            )

            if should_merge:
                left = min(cx1, x1)
                top = min(cy1, y1)
                right = max(cx2, x2)
                bottom = max(cy2, y2)
                current["bbox"] = [left, top, right, bottom]
                current["text"] = f"{current.get('text', '')}{joiner}{item.get('text', '')}".strip()
            else:
                merged.append(current)
                current = dict(item)

            prev_small = is_small

        if current is not None:
            merged.append(current)
        return merged

    def run(self):
        ai_start = time.perf_counter()
        worker = OllamaTranslator(self.config)

        # ══ 贴字翻译模式 ══
        if self.config.get("use_overlay_ocr", False):
            try:
                items = worker.run_ocr_with_coords(self.image_bytes)
                if items:
                    if self.config.get("overlay_auto_merge_lines", False):
                        items = self._merge_overlay_lines(
                            items,
                            int(self.config.get("overlay_min_line_height", 40)),
                            int(self.config.get("overlay_max_line_gap", 4)),
                            self.config.get("overlay_joiner", " "),
                            self.config.get("line_start_chars", ",.;:!?)]}、，。！？；：」』）】》"),
                            self.config.get("line_end_chars", ".!?。！？…"),
                        )
                    # 发出原文供 OCR 标签显示（合并所有文本）
                    self.ocr_ready.emit("\n".join(it["text"] for it in items))
                    if self.config.get("use_llm", True):
                        items = worker.run_llm_batch(items)
                    else:
                        for it in items:
                            it["translated"] = it["text"]
                    self.overlay_ready.emit(items)
                    self.finished.emit("", time.perf_counter() - ai_start)
                    return
                # 模型不支持贴字格式 → 降级到普通模式，继续往下走
                print("[overlay_ocr] 模型未返回坐标格式，降级为普通 OCR")
            except Exception as e:
                print(f"[overlay_ocr error] {e}")

        # ══ 普通翻译模式 ══
        ocr_text = ""
        final_text = ""
        try:
            if self.config.get("use_ocr", True):
                ocr_text = worker.run_ocr(self.image_bytes)
                self.ocr_ready.emit(ocr_text)

            if self.config.get("use_llm", True):
                img = None if ocr_text else self.image_bytes
                if self.config.get("use_stream", False):
                    cumulative = ""
                    for chunk in worker.run_llm_stream(ocr_text, img):
                        cumulative += chunk
                        self.partial_text.emit(cumulative)
                    final_text = cumulative
                else:
                    final_text = worker.run_llm(ocr_text, img)
            else:
                final_text = ocr_text

        except Exception as e:
            final_text = f"Error: {e}"

        self.finished.emit(final_text, time.perf_counter() - ai_start)


# ══════════════════════════════════════════════
# 带描边的彩色文字标签
# ══════════════════════════════════════════════

class OutlinedLabel(QLabel):
    def __init__(self, text="", parent=None, color="#ffffff", margins=(4, 4, 4, 4)):
        super().__init__(text, parent)
        self._text_color = QColor(color)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setContentsMargins(*margins)

    def setColor(self, hex_color: str):
        self._text_color = QColor(hex_color)
        self.update()

    def paintEvent(self, event):
        if not self.text():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self.font())

        rect = self.contentsRect()
        flags = int(self.alignment()) | Qt.TextWordWrap

        # 黑色描边（8 方向偏移）
        painter.setPen(QPen(QColor(0, 0, 0, 230)))
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1)]:
            painter.drawText(rect.translated(dx, dy), flags, self.text())

        # 主体文字
        painter.setPen(self._text_color)
        painter.drawText(rect, flags, self.text())
        painter.end()


# ══════════════════════════════════════════════
# 贴字翻译：屏幕透明悬浮层
# ══════════════════════════════════════════════

class TextOverlayWindow(QWidget):
    """
    全屏透明贴字翻译窗口。
    覆盖目标屏幕，在对应位置显示翻译后的文本块，鼠标事件穿透。

    坐标约定：
      capture_region 存储的是【屏幕内逻辑像素】偏移（RegionSelector 的框选结果）。
      OCR bbox 为 0-1000 归一化坐标，相对于截图区域。
      两者组合即可得到文本在本窗口内的逻辑坐标。
    """

    def __init__(self, screen, capture_region: dict, cfg: dict):
        super().__init__()
        self._screen = screen
        self._region = capture_region   # {"x","y","w","h"} 屏幕内逻辑像素
        self._cfg    = cfg
        self._labels: list[OutlinedLabel] = []

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint |
            Qt.Tool | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setGeometry(screen.geometry())   # 覆盖整块屏幕
        self.show()

    # ── 更新贴字内容 ──

    @staticmethod
    def _fit_font_size(text: str, width: int, height: int, min_size=8, max_size=54) -> int:
        """在给定矩形内自适应字体大小。"""
        if not text.strip() or width <= 4 or height <= 4:
            return min_size

        low, high = min_size, max_size
        best = min_size
        while low <= high:
            mid = (low + high) // 2
            metrics = QFontMetrics(QFont("Microsoft YaHei UI", mid))
            text_rect = metrics.boundingRect(0, 0, width, height,
                                             Qt.TextWordWrap | Qt.AlignCenter,
                                             text)
            if text_rect.height() <= height and text_rect.width() <= width:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _map_bbox_to_overlay(self, bbox: list, ocr_size: tuple | None) -> tuple[int, int, int, int]:
        """将模型 bbox 映射为当前屏幕浮层中的逻辑坐标。"""
        rx, ry = self._region["x"], self._region["y"]
        rw, rh = self._region["w"], self._region["h"]
        x1, y1, x2, y2 = bbox

        # deepseek 常见两种格式：
        # 1) 0-1000 归一化坐标
        # 2) 基于输入图像尺寸的像素坐标
        if max(abs(v) for v in (x1, y1, x2, y2)) <= 1000:
            fx1, fy1 = x1 / 1000.0, y1 / 1000.0
            fx2, fy2 = x2 / 1000.0, y2 / 1000.0
        elif ocr_size and ocr_size[0] > 0 and ocr_size[1] > 0:
            fx1, fy1 = x1 / ocr_size[0], y1 / ocr_size[1]
            fx2, fy2 = x2 / ocr_size[0], y2 / ocr_size[1]
        else:
            fx1, fy1 = 0.0, 0.0
            fx2, fy2 = 1.0, 1.0

        lx1 = int(rx + fx1 * rw)
        ly1 = int(ry + fy1 * rh)
        lx2 = int(rx + fx2 * rw)
        ly2 = int(ry + fy2 * rh)
        return lx1, ly1, lx2, ly2

    def update_items(self, items: list, ocr_size: tuple | None = None):
        """
        items: [{"text": str, "translated": str, "bbox": [x1,y1,x2,y2]}, ...]
        bbox 为 0-1000 归一化坐标，原点为截图区域左上角。
        """
        for lbl in self._labels:
            lbl.deleteLater()
        self._labels.clear()

        trans_color = self._cfg.get("trans_color", "#FFFFFF")
        show_orig   = self._cfg.get("show_ocr_text", False)
        ocr_color   = self._cfg.get("ocr_color", "#FFFF88")
        min_box_h   = int(self._cfg.get("overlay_min_box_height", 28))
        show_debug_boxes = self._cfg.get("show_overlay_debug_boxes", False)

        for item in items:
            bbox        = item["bbox"]           # [x1, y1, x2, y2] 0-1000
            orig_text   = item.get("text", "")
            trans_text  = item.get("translated", orig_text)

            lx1, ly1, lx2, ly2 = self._map_bbox_to_overlay(bbox, ocr_size)
            lw = max(lx2 - lx1, 10)
            lh = max(ly2 - ly1, min_box_h)

            # 仅调试时显示模型原始框
            if show_debug_boxes:
                debug_box = QLabel(self)
                debug_box.setGeometry(lx1, ly1, lw, lh)
                debug_box.setStyleSheet("border: 2px solid rgba(255, 80, 80, 220); background: transparent;")
                debug_box.show()

            # 背景块（深色半透明遮罩）
            bg = QLabel(self)
            bg_h = lh * 2 if show_orig else lh
            bg.setGeometry(lx1, ly1, lw, bg_h)
            bg.setStyleSheet("background: rgba(8,8,8,200); border-radius: 4px;")
            bg.show()
            # bg 作为普通 QLabel 不加入 _labels 列表（不需要颜色更新），
            # 但需要随本窗口清理 → 用父子关系自动管理

            # 若开启"显示原文"，在方块上方叠一行小字原文
            if show_orig and orig_text:
                orig_lbl = OutlinedLabel(orig_text, self, color=ocr_color, margins=(1, 0, 1, 0))
                orig_font = self._fit_font_size(orig_text, lw - 8, max(lh - 8, 10),
                                                min_size=8, max_size=max(10, int(lh * 0.9)))
                orig_lbl.setFont(QFont("Microsoft YaHei UI", orig_font))
                orig_lbl.setWordWrap(True)
                orig_lbl.setAlignment(Qt.AlignCenter)
                orig_lbl.setGeometry(lx1, ly1, lw, lh)
                orig_lbl.show()
                self._labels.append(orig_lbl)

            # 译文
            trans_lbl = OutlinedLabel(trans_text, self, color=trans_color, margins=(1, 0, 1, 0))
            trans_h = lh if show_orig else lh
            trans_y = ly1 + (lh if show_orig else 0)
            max_font = max(10, int(trans_h * (1.2 if show_orig else 1.0)))
            font_size = self._fit_font_size(trans_text, lw - 8, max(trans_h - 8, 10),
                                            min_size=8, max_size=max_font)
            trans_lbl.setFont(QFont("Microsoft YaHei UI", font_size))
            trans_lbl.setWordWrap(True)
            trans_lbl.setAlignment(Qt.AlignCenter)
            trans_lbl.setGeometry(lx1, trans_y, lw, trans_h)
            trans_lbl.show()
            self._labels.append(trans_lbl)

        self.update()

    def clear_items(self):
        for lbl in self._labels:
            lbl.deleteLater()
        self._labels.clear()
        # 清理所有子 widget（包括背景 QLabel）
        for child in self.findChildren(QLabel):
            child.deleteLater()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))   # 完全透明
        painter.end()


# ══════════════════════════════════════════════
# 全屏区域框选工具
# ══════════════════════════════════════════════

class RegionSelector(QWidget):
    """
    单屏半透明框选遮罩。
    只覆盖传入的那块 QScreen，彻底规避跨屏 DPI 问题。

    坐标系说明：
      - widget 内 e.pos() 是相对该屏左上角的**逻辑像素**
      - 乘以 screen.devicePixelRatio() 得到**屏幕相对物理像素**
      - grabWindow(0) 在同一块屏上拍出的图片坐标系完全吻合
    """
    # x, y, w, h —— 相对目标屏幕左上角的物理像素
    region_selected = Signal(int, int, int, int)
    closed          = Signal()

    def __init__(self, screen):
        super().__init__()
        self._screen = screen
        self._dpr    = screen.devicePixelRatio()

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setCursor(Qt.CrossCursor)

        # 精确覆盖该屏（逻辑坐标），不涉及其他屏
        self.setGeometry(screen.geometry())
        self.show()

        self._p1 = QPoint()   # widget 内逻辑坐标
        self._p2 = QPoint()
        self._selecting = False

    def _to_local(self, logical_local: QPoint) -> QPoint:
        """返回屏幕内逻辑坐标（原样保留，实际 mss 坐标在 capture_task 中计算）"""
        return logical_local

    def _sel_rect_logical(self) -> QRect:
        return QRect(self._p1, self._p2).normalized()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 130))

        if not self._p1.isNull() and not self._p2.isNull():
            sel = self._sel_rect_logical()

            # 镂空选区
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(sel, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            # 蓝色边框
            painter.setPen(QPen(QColor(40, 140, 255), 2))
            painter.drawRect(sel)

            # 逻辑像素提示（存储值，与屏幕内坐标一致）
            hint = f"{sel.width()} × {sel.height()} logical px  |  Esc 取消"
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 11))
            tip = sel.bottomRight() + QPoint(8, -4)
            if tip.x() + 220 > self.width():
                tip = sel.bottomLeft() + QPoint(-230, -4)
            painter.drawText(tip, hint)

        painter.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._p1 = self._p2 = e.pos()
            self._selecting = True

    def mouseMoveEvent(self, e):
        if self._selecting:
            self._p2 = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._selecting:
            self._p2 = e.pos()
            self._selecting = False
            sel = self._sel_rect_logical()
            if sel.width() > 5 and sel.height() > 5:
                tl = self._to_local(sel.topLeft())
                br = self._to_local(sel.bottomRight())
                self.region_selected.emit(tl.x(), tl.y(),
                                          br.x() - tl.x(), br.y() - tl.y())
            self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, e):
        self.closed.emit()
        super().closeEvent(e)


# ══════════════════════════════════════════════
# 悬浮翻译浮窗
# ══════════════════════════════════════════════

class SubtitleOverlay(QWidget):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.is_processing = False
        self.worker_thread = None
        self.step1_duration = 0
        self._bottom_anchor = None   # 用于"向上扩展"模式固定底部

        self.mouse_listener = None
        self.key_listener   = None
        self.input_signal   = InputSignal()
        self.input_signal.triggered.connect(self.capture_task)
        self.last_trigger_time = 0
        self.text_overlay: TextOverlayWindow | None = None  # 贴字翻译浮层
        self._latest_ocr_image_size: tuple[int, int] | None = None

        self.update_window_flags()

        # ── 布局 ──
        fixed_w = int(self.cfg.get("ui_max_width", 800))
        self.setFixedWidth(fixed_w + 28)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.main_layout.setContentsMargins(14, 10, 14, 10)
        self.main_layout.setSpacing(4)

        # OCR 原文标签
        self.ocr_label = OutlinedLabel("", self, color=self.cfg.get("ocr_color", "#FFFF88"))
        self.ocr_label.setFont(QFont("Microsoft YaHei UI", 14))
        self.ocr_label.setWordWrap(True)
        self.ocr_label.setFixedWidth(fixed_w)
        self.ocr_label.setAlignment(Qt.AlignCenter)
        self.ocr_label.setVisible(False)  # 初始隐藏

        # 译文标签
        self.trans_label = OutlinedLabel("正在等待截图...", self, color=self.cfg.get("trans_color", "#FFFFFF"))
        self.trans_label.setFont(QFont("Microsoft YaHei UI", 18))
        self.trans_label.setWordWrap(True)
        self.trans_label.setFixedWidth(fixed_w)
        self.trans_label.setAlignment(Qt.AlignCenter)

        self.main_layout.addWidget(self.ocr_label)
        self.main_layout.addWidget(self.trans_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_task)
        self.apply_mode()

    # ── 窗口属性 ──

    def update_window_flags(self):
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.cfg.get("always_on_top", True):
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        if self.cfg.get("window_visible", True):
            self.show()
        else:
            self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 12, 12)
        painter.fillPath(path, QColor(10, 10, 10, 210))
        painter.end()

    def showEvent(self, event):
        super().showEvent(event)
        if self._bottom_anchor is None:
            self._bottom_anchor = self.y() + self.height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 向上扩展：固定底部边缘，向上伸长
        if (self.cfg.get("grow_direction", "up") == "up"
                and self._bottom_anchor is not None):
            self.move(self.x(), self._bottom_anchor - self.height())

    # ── 模式控制 ──

    def apply_mode(self):
        self.timer.stop()
        self.stop_listeners()
        mode = self.cfg.get("capture_mode", "interval")
        if mode == "interval":
            self.timer.start(int(self.cfg.get("capture_interval", 2.5) * 1000))
        elif mode == "trigger":
            key_name = self.cfg.get("trigger_key", "Left Click")
            if "Click" in key_name:
                self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
                self.mouse_listener.start()
            else:
                self.key_listener = keyboard.Listener(on_release=self.on_key_release)
                self.key_listener.start()

    def stop_listeners(self):
        if self.mouse_listener: self.mouse_listener.stop(); self.mouse_listener = None
        if self.key_listener:   self.key_listener.stop();   self.key_listener   = None

    def on_mouse_click(self, x, y, button, pressed):
        target = mouse.Button.left if self.cfg["trigger_key"] == "Left Click" else mouse.Button.right
        if pressed and button == target:
            self.trigger_signal()

    def on_key_release(self, key):
        try:
            t = self.cfg["trigger_key"]
            if t == "Space" and key == keyboard.Key.space: self.trigger_signal()
            elif t == "Enter" and key == keyboard.Key.enter: self.trigger_signal()
        except: pass

    def trigger_signal(self):
        if time.time() - self.last_trigger_time > 1.0:
            self.last_trigger_time = time.time()
            self.input_signal.triggered.emit()

    # ── 布局热更新 ──

    def update_layout_settings(self):
        fixed_w = int(self.cfg.get("ui_max_width", 800))
        self.setFixedWidth(fixed_w + 28)
        for lbl in (self.ocr_label, self.trans_label):
            lbl.setFixedWidth(fixed_w)
        self.ocr_label.setColor(self.cfg.get("ocr_color", "#FFFF88"))
        self.trans_label.setColor(self.cfg.get("trans_color", "#FFFFFF"))
        self.ocr_label.setVisible(
            self.cfg.get("show_ocr_text", False) and bool(self.ocr_label.text())
        )
        self._adjust_size()

    def _get_capture_screen(self):
        name = self.cfg.get("capture_screen_name", "")
        for s in QApplication.screens():
            if s.name() == name:
                return s
        return QApplication.primaryScreen()

    @staticmethod
    def _region_to_mss_rect(region: dict, screen_name: str) -> dict:
        """
        将框选器存储的【屏幕内逻辑坐标】转换为 mss 物理坐标。

        做法：
          1. 通过物理尺寸匹配，从 mss 枚举的显示器列表中找到对应的条目，
             直接读取其 left/top（mss 自己的物理原点，100% 准确）。
          2. 逻辑偏移 × 该屏 DPR → 物理偏移，加到原点上。

        这样彻底绕开 QScreen.geometry() 在不同 DPI-Awareness 模式下
        返回值含义不一致的问题。
        """
        import mss

        # 找到目标 QScreen
        target = None
        for s in QApplication.screens():
            if s.name() == screen_name:
                target = s
                break
        if target is None:
            target = QApplication.primaryScreen()

        dpr = target.devicePixelRatio()

        # 计算目标屏的物理像素尺寸，用于匹配 mss 显示器
        # QScreen.size() 在 Windows 上通常等于逻辑尺寸；乘以 DPR 得物理尺寸
        phys_w = round(target.size().width()  * dpr)
        phys_h = round(target.size().height() * dpr)

        with mss.mss() as sct:
            real_monitors = sct.monitors[1:]   # [0] 是虚拟桌面整体

            # 按坐标排序 Qt 屏幕和 mss 显示器，用相对顺序匹配
            # （即使 Qt 与 mss 坐标单位不同，左→右/上→下的顺序是一致的）
            sorted_qt  = sorted(QApplication.screens(),
                                key=lambda s: (s.geometry().x(), s.geometry().y()))
            sorted_mss = sorted(real_monitors,
                                key=lambda m: (m["left"], m["top"]))

            mss_mon = None

            # 优先用屏幕名称 + 位置顺序匹配
            try:
                qt_idx  = sorted_qt.index(target)
                if qt_idx < len(sorted_mss):
                    candidate = sorted_mss[qt_idx]
                    # 物理尺寸校验（容差 ±8px，应对分辨率舍入）
                    if (abs(candidate["width"]  - phys_w) <= 8 and
                            abs(candidate["height"] - phys_h) <= 8):
                        mss_mon = candidate
            except ValueError:
                pass

            # 退路：遍历找第一个尺寸吻合的
            if mss_mon is None:
                for m in real_monitors:
                    if abs(m["width"] - phys_w) <= 8 and abs(m["height"] - phys_h) <= 8:
                        mss_mon = m
                        break

            # 最终退路：主显示器
            if mss_mon is None:
                mss_mon = sct.monitors[1]

            rx = mss_mon["left"] + int(region["x"] * dpr)
            ry = mss_mon["top"]  + int(region["y"] * dpr)
            rw = int(region["w"] * dpr)
            rh = int(region["h"] * dpr)
            return {"left": rx, "top": ry, "width": rw, "height": rh,
                    "_phys_w": rw, "_phys_h": rh}

    # ── 截图主流程 ──

    def _reset_text_overlay_before_capture(self):
        """
        下一次截图前先清掉旧贴字浮层，避免模型反复识别自己上一次贴出来的文本。
        """
        if self.text_overlay and isValid(self.text_overlay):
            self.text_overlay.clear_items()
            self.text_overlay.close()
            self.text_overlay = None
            QApplication.processEvents()

    def capture_task(self):
        if self.is_processing: return

        step1_start = time.perf_counter()
        try:
            # 贴字模式下，截图前先移除已有贴字窗口，防止出现循环识别
            if self.cfg.get("use_overlay_ocr", False):
                self._reset_text_overlay_before_capture()

            was_visible = self.isVisible()
            should_hide = self.cfg.get("auto_hide", True)
            if should_hide and was_visible:
                self.setVisible(False)
                QApplication.processEvents()
                time.sleep(0.02)

            # ── 确定截图范围（mss 物理坐标）──
            source = self.cfg.get("capture_source", "window")
            import mss
            from PySide6.QtGui import QImage, QPixmap

            if source == "region":
                region = self.cfg.get("capture_region")
                if not region:
                    if should_hide and was_visible: self.setVisible(True)
                    return
                screen_name = self.cfg.get("capture_screen_name", "")
                mss_rect = self._region_to_mss_rect(region, screen_name)
                x, y, w, h = (mss_rect["left"], mss_rect["top"],
                               mss_rect["_phys_w"], mss_rect["_phys_h"])
            else:
                hwnd = self.cfg.get("target_hwnd", 0)
                if not hwnd:
                    if should_hide and was_visible: self.setVisible(True)
                    return
                x, y, w, h = get_window_rect(hwnd)
                mss_rect = {"left": x, "top": y, "width": w, "height": h}

            # ── 用 mss 截图（支持跨显示器，支持硬件加速窗口）──
            with mss.mss() as sct:
                shot = sct.grab(mss_rect)

            # 截图完成后再恢复显示
            if should_hide and was_visible:
                self.setVisible(True)

            # mss 返回 BGRA，转为 QImage → QPixmap
            img = QImage(
                shot.raw, shot.width, shot.height,
                shot.width * 4, QImage.Format.Format_ARGB32
            )
            pix = QPixmap.fromImage(img)

            if pix.isNull():
                self.is_processing = False
                return

            scale = self.cfg.get("scale_factor", 0.5)
            if scale < 1.0:
                pix = pix.scaled(
                    int(w * scale), int(h * scale),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

            self._latest_ocr_image_size = (pix.width(), pix.height())

            img_format = str(self.cfg.get("ocr_image_format", "PNG")).upper()
            if img_format not in ("PNG", "JPEG", "JPG"):
                img_format = "PNG"

            # OCR 请求与调试文件使用同一份编码字节，避免二次压缩带来的识别差异。
            buf = QBuffer()
            buf.open(QIODevice.WriteOnly)
            if img_format in ("JPEG", "JPG"):
                jpeg_q = int(self.cfg.get("ocr_image_quality", 95))
                jpeg_q = max(1, min(100, jpeg_q))
                pix.save(buf, "JPEG", jpeg_q)
                debug_path = "debug_current_vision.jpg"
            else:
                pix.save(buf, "PNG")
                debug_path = "debug_current_vision.png"

            img_bytes = bytes(buf.data())
            buf.close()

            with open(debug_path, "wb") as f:
                f.write(img_bytes)

            self.step1_duration = time.perf_counter() - step1_start
            self.is_processing  = True

            # ── 贴字模式：确保浮层窗口已创建 ──
            if self.cfg.get("use_overlay_ocr", False):
                region = self.cfg.get("capture_region")
                if region and (self.text_overlay is None or not isValid(self.text_overlay)):
                    screen = self._get_capture_screen()
                    self.text_overlay = TextOverlayWindow(screen, region, self.cfg)

            self.worker_thread = TranslationThread(img_bytes, self.cfg)
            self.worker_thread.ocr_ready.connect(self.on_ocr_ready)
            self.worker_thread.partial_text.connect(self.on_partial_text)
            self.worker_thread.finished.connect(self.on_translated)
            self.worker_thread.overlay_ready.connect(self.on_overlay_ready)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            self.worker_thread.start()

        except Exception as e:
            print(f"[capture_task error] {e}")
            self.is_processing = False
            if self.cfg.get("auto_hide", True): self.setVisible(True)

    # ── 翻译结果回调 ──

    def on_ocr_ready(self, text: str):
        """OCR 阶段完成，展示原文（若设置开启）"""
        print(f"[OCR 输出全文]\n{text}\n[OCR 输出结束]")
        clean_text = self._post_process_text(text)
        if self.cfg.get("show_ocr_text", False) and clean_text.strip():
            self.ocr_label.setText(clean_text)
            self.ocr_label.setVisible(True)
            self._adjust_size()

    def on_overlay_ready(self, items: list):
        """贴字翻译：将带坐标的译文更新到屏幕浮层"""
        if self.text_overlay and isValid(self.text_overlay):
            self.text_overlay.update_items(items, self._latest_ocr_image_size)
        overlay_lines = []
        for idx, it in enumerate(items, start=1):
            src = it.get("text", "")
            dst = it.get("translated", src)
            overlay_lines.append(f"{dst}")
        if overlay_lines:
            debug_text = "\n\n".join(overlay_lines)
            print(f"[贴字模型输出全文]\n{debug_text}\n[贴字模型输出结束]")
            self.trans_label.setText(self._post_process_text(debug_text))
            self._adjust_size()

    def on_partial_text(self, text: str):
        """流式输出：累计更新译文"""
        processed = self._post_process_text(text)
        if processed.strip():
            self.trans_label.setText(processed)
            self._adjust_size()

    def on_translated(self, text: str, ai_duration: float):
        total = self.step1_duration + ai_duration
        print(f"\n{'='*40}\n时戳: {time.strftime('%H:%M:%S')}")
        print(f"截图: {self.step1_duration:.3f}s  |  AI: {ai_duration:.3f}s  |  总计: {total:.3f}s")
        print(f"模型输出全文:\n{text}\n{'='*40}")

        processed = self._post_process_text(text)
        if processed.strip():
            self.trans_label.setText(processed)
            if self.cfg.get("auto_copy"):
                QGuiApplication.clipboard().setText(processed)
            self._adjust_size()

        self.is_processing = False

    @staticmethod
    def _remove_blank_lines(text: str) -> str:
        lines = text.splitlines()
        return "\n".join(line for line in lines if line.strip())

    def _post_process_text(self, text: str) -> str:
        result = text or ""
        if self.cfg.get("remove_blank_lines", False):
            result = self._remove_blank_lines(result)
        return result

    def _adjust_size(self):
        """调整大小前锁定底部锚点（向上扩展模式）"""
        if self.cfg.get("grow_direction", "up") == "up":
            self._bottom_anchor = self.y() + self.height()
        self.adjustSize()

    # ── 拖拽移动 ──

    def mousePressEvent(self, e):
        self._drag_pos = e.globalPosition().toPoint()

    def mouseMoveEvent(self, e):
        delta = e.globalPosition().toPoint() - self._drag_pos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self._drag_pos = e.globalPosition().toPoint()
        # 拖拽时更新底部锚点，防止释放后跳回
        self._bottom_anchor = self.y() + self.height()

    def closeEvent(self, e):
        self.timer.stop()
        self.stop_listeners()
        if self.worker_thread and isValid(self.worker_thread):
            self.worker_thread.terminate()
        if self.text_overlay and isValid(self.text_overlay):
            self.text_overlay.close()
        e.accept()


# ══════════════════════════════════════════════
# 设置界面
# ══════════════════════════════════════════════

class SettingInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("settingInterface")
        self.view = QWidget()
        self.layout = QVBoxLayout(self.view)
        self.layout.setContentsMargins(30, 20, 30, 20)
        self.layout.setSpacing(15)

        # ── 截图触发策略 ──
        self.mode_group = SettingCardGroup("截图触发策略", self.view)

        self.mode_card = CustomSettingCard(FIF.GAME, "触发模式", "选择自动截图或按键触发", self.mode_group)
        self.mode_seg = SegmentedWidget(self.view)
        self.mode_seg.addItem("interval", "定时自动")
        self.mode_seg.addItem("trigger",  "按键触发")
        self.mode_card.addWidget(self.mode_seg)

        self.interval_card = CustomSettingCard(FIF.HISTORY, "定时截图间隔", "单位：秒", self.mode_group)
        self.interval_spin = DoubleSpinBox()
        self.interval_spin.setRange(0.5, 10.0); self.interval_spin.setSingleStep(0.5)
        self.interval_card.addWidget(self.interval_spin)

        self.trigger_card = CustomSettingCard(FIF.TAG, "触发按键", "在游戏中按下此键时翻译", self.mode_group)
        self.trigger_combo = ComboBox()
        self.trigger_combo.addItems(["Left Click", "Right Click", "Space", "Enter"])
        self.trigger_card.addWidget(self.trigger_combo)

        self.mode_group.addSettingCard(self.mode_card)
        self.mode_group.addSettingCard(self.interval_card)
        self.mode_group.addSettingCard(self.trigger_card)
        self.layout.addWidget(self.mode_group)

        # ── 窗口行为 ──
        self.win_group = SettingCardGroup("窗口行为", self.view)

        self.width_card = CustomSettingCard(FIF.ALIGNMENT, "字幕固定宽度", "文本框宽度固定为此值 (px)", self.win_group)
        self.width_spin = DoubleSpinBox()
        self.width_spin.setRange(300, 1920); self.width_spin.setSingleStep(50)
        self.width_card.addWidget(self.width_spin)

        self.grow_card = CustomSettingCard(FIF.SCROLL, "文本框扩展方向", "文字增多时向哪个方向延伸", self.win_group)
        self.grow_seg = SegmentedWidget(self.view)
        self.grow_seg.addItem("up",   "向上扩展")
        self.grow_seg.addItem("down", "向下扩展")
        self.grow_card.addWidget(self.grow_seg)

        self.hide_card = CustomSettingCard(FIF.HIDE, "截图时隐藏自身", "防止字幕遮挡原文（推荐）", self.win_group)
        self.sw_hide = SwitchButton()
        self.hide_card.addWidget(self.sw_hide)

        self.visible_card = CustomSettingCard(FIF.VIEW, "显示翻译窗口", "关闭则隐藏悬浮字幕", self.win_group)
        self.sw_visible = SwitchButton()
        self.visible_card.addWidget(self.sw_visible)

        self.top_card = CustomSettingCard(FIF.PIN, "窗口置顶显示", "保持悬浮窗在最上层", self.win_group)
        self.sw_top = SwitchButton()
        self.top_card.addWidget(self.sw_top)

        for card in (self.width_card, self.grow_card, self.hide_card,
                     self.visible_card, self.top_card):
            self.win_group.addSettingCard(card)
        self.layout.addWidget(self.win_group)

        # ── 性能与策略 ──
        self.perf_group = SettingCardGroup("性能与策略", self.view)

        self.scale_card = CustomSettingCard(FIF.ZOOM, "截图缩放比例", "调小可提升 AI 响应速度", self.perf_group)
        self.scale_spin = DoubleSpinBox()
        self.scale_spin.setRange(0.1, 1.0); self.scale_spin.setSingleStep(0.1)
        self.scale_card.addWidget(self.scale_spin)

        self.stream_card = CustomSettingCard(FIF.SPEED_HIGH, "流式输出", "逐字显示翻译结果（体感更快）", self.perf_group)
        self.sw_stream = SwitchButton()
        self.stream_card.addWidget(self.sw_stream)

        self.ocr_sw_card = CustomSettingCard(FIF.CAMERA, "启用 OCR 文字提取", "关闭则直接发送截图给 LLM", self.perf_group)
        self.sw_ocr = SwitchButton()
        self.ocr_sw_card.addWidget(self.sw_ocr)

        self.llm_sw_card = CustomSettingCard(FIF.EDIT, "启用智能翻译润色", "关闭则仅显示原始 OCR 内容", self.perf_group)
        self.sw_llm = SwitchButton()
        self.llm_sw_card.addWidget(self.sw_llm)

        self.copy_sw_card = CustomSettingCard(FIF.COPY, "自动同步剪贴板", "翻译结果自动复制", self.perf_group)
        self.sw_copy = SwitchButton()
        self.copy_sw_card.addWidget(self.sw_copy)

        self.remove_blank_card = CustomSettingCard(
            FIF.ALIGNMENT,
            "自动去除空行",
            "移除翻译文本框中的空白行，避免显示大段间隙",
            self.perf_group
        )
        self.sw_remove_blank = SwitchButton()
        self.remove_blank_card.addWidget(self.sw_remove_blank)

        for card in (self.scale_card, self.stream_card, self.ocr_sw_card,
                     self.llm_sw_card, self.copy_sw_card, self.remove_blank_card):
            self.perf_group.addSettingCard(card)
        self.layout.addWidget(self.perf_group)

        self.layout.addStretch(1)

        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: transparent; border: none;")

        # 触发模式联动
        self.mode_seg.currentItemChanged.connect(self.on_mode_changed)

    def on_mode_changed(self, k):
        self.interval_card.setVisible(k == "interval")
        self.trigger_card.setVisible(k == "trigger")


# ══════════════════════════════════════════════
# 主页（含区域框选）
# ══════════════════════════════════════════════

class HomeInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("homeInterface")
        self.view = QWidget()
        self.layout = QVBoxLayout(self.view)
        self.layout.setContentsMargins(35, 30, 35, 30)
        self.layout.setSpacing(20)
        self.layout.addWidget(SubtitleLabel("控制台", self.view))

        # ── 截图来源（窗口 vs 区域，互斥）──
        self.source_card = CustomSettingCard(FIF.PHOTO, "截图来源", "选择截取目标窗口还是手动框选区域", self.view)
        self.source_seg = SegmentedWidget(self.view)
        self.source_seg.addItem("window", "窗口截图")
        self.source_seg.addItem("region", "区域框选")
        self.source_seg.setFixedWidth(220)
        self.source_card.addWidget(self.source_seg)
        self.layout.addWidget(self.source_card)

        # 游戏窗口选择
        self.window_card = CustomSettingCard(FIF.APPLICATION, "游戏窗口", "选择翻译目标（窗口模式）", self.view)
        self.combo = ComboBox(self.window_card)
        self.combo.setFixedWidth(250)
        self.window_card.addWidget(self.combo)
        self.layout.addWidget(self.window_card)

        # 目标屏幕选择
        self.screen_card = CustomSettingCard(FIF.FIT_PAGE, "截图目标屏幕", "区域框选模式下从哪块屏幕截图", self.view)
        self.screen_combo = ComboBox(self.screen_card)
        self.screen_combo.setFixedWidth(250)
        self._populate_screens()
        self.screen_card.addWidget(self.screen_combo)
        self.layout.addWidget(self.screen_card)

        # 截图区域
        self.region_card = CustomSettingCard(FIF.ZOOM, "截图区域", "全窗口（未框选）", self.view)
        region_btns = QWidget()
        rbl = QHBoxLayout(region_btns)
        rbl.setContentsMargins(0, 0, 0, 0); rbl.setSpacing(6)
        self.region_btn       = PushButton(FIF.PENCIL_INK, "框选区域", region_btns)
        self.clear_region_btn = PushButton(FIF.DELETE,     "清除",     region_btns)
        self.clear_region_btn.setEnabled(False)
        rbl.addWidget(self.region_btn)
        rbl.addWidget(self.clear_region_btn)
        self.region_card.addWidget(region_btns)
        self.layout.addWidget(self.region_card)

        # 启停按钮
        self.btn_layout = QHBoxLayout()
        self.refresh_btn = PushButton(FIF.SYNC, "刷新窗口列表", self.view)
        self.btn_layout.addWidget(self.refresh_btn)
        self.layout.addLayout(self.btn_layout)
        self.layout.addStretch(1)

        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: transparent; border: none;")

        # 截图来源联动
        self.source_seg.currentItemChanged.connect(self.on_source_changed)
        self.on_source_changed("window")  # 初始状态

    def on_source_changed(self, key):
        is_window = (key == "window")
        self.window_card.setVisible(is_window)
        self.refresh_btn.setVisible(is_window)
        self.screen_card.setVisible(not is_window)
        self.region_card.setVisible(not is_window)

    def _populate_screens(self):
        self.screen_combo.clear()
        for i, s in enumerate(QApplication.screens()):
            sz = s.size()   # 物理分辨率
            label = f"屏幕 {i+1}  {sz.width()}×{sz.height()}  ({s.name()})"
            self.screen_combo.addItem(label, userData=s.name())

    def selected_screen(self):
        """返回当前选中的 QScreen，找不到则返回主屏"""
        name = self.screen_combo.currentData()
        for s in QApplication.screens():
            if s.name() == name:
                return s
        return QApplication.primaryScreen()


class AISettingInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("aiSettingInterface")
        self.view = QWidget()
        self.layout = QVBoxLayout(self.view)
        self.layout.setContentsMargins(30, 20, 30, 20)
        self.layout.setSpacing(15)

        self.ocr_group = SettingCardGroup("OCR 配置", self.view)
        self.ocr_model_card = CustomSettingCard(FIF.CODE, "OCR 识别模型", "视觉模型，用于提取图片文字", self.ocr_group)
        self.ocr_model_edit = LineEdit()
        self.ocr_model_card.addWidget(self.ocr_model_edit)

        self.ocr_api_card = CustomSettingCard(FIF.LINK, "OCR API", "默认本地 Ollama /api/generate", self.ocr_group)
        self.ocr_api_edit = LineEdit()
        self.ocr_api_card.addWidget(self.ocr_api_edit)

        self.ocr_key_card = CustomSettingCard(FIF.FINGERPRINT, "OCR API Key", "默认空，留空则不附带 Authorization", self.ocr_group)
        self.ocr_key_edit = LineEdit()
        self.ocr_key_card.addWidget(self.ocr_key_edit)

        self.ocr_ctx_card = CustomSettingCard(FIF.DICTIONARY, "OCR 上下文长度", "请求参数 num_ctx（0 表示使用模型默认）", self.ocr_group)
        self.ocr_ctx_spin = DoubleSpinBox()
        self.ocr_ctx_spin.setRange(0, 131072)
        self.ocr_ctx_spin.setSingleStep(512)
        self.ocr_ctx_spin.setDecimals(0)
        self.ocr_ctx_card.addWidget(self.ocr_ctx_spin)

        self.ocr_prompt_card = PromptSettingCard(
            FIF.CAMERA, "OCR 提示词", "普通 OCR 指令（贴字模式不使用）", 90, self.ocr_group
        )
        self.ocr_prompt_edit = self.ocr_prompt_card.editor

        for card in (self.ocr_model_card, self.ocr_api_card, self.ocr_key_card, self.ocr_ctx_card, self.ocr_prompt_card):
            self.ocr_group.addSettingCard(card)
        self.layout.addWidget(self.ocr_group)

        self.llm_group = SettingCardGroup("LLM 配置", self.view)
        self.llm_model_card = CustomSettingCard(FIF.CHAT, "LLM 翻译模型", "语言模型，用于文本润色", self.llm_group)
        self.llm_model_edit = LineEdit()
        self.llm_model_card.addWidget(self.llm_model_edit)

        self.llm_api_card = CustomSettingCard(FIF.LINK, "LLM API", "默认本地 Ollama /api/generate", self.llm_group)
        self.llm_api_edit = LineEdit()
        self.llm_api_card.addWidget(self.llm_api_edit)

        self.llm_key_card = CustomSettingCard(FIF.FINGERPRINT, "LLM API Key", "默认空，留空则不附带 Authorization", self.llm_group)
        self.llm_key_edit = LineEdit()
        self.llm_key_card.addWidget(self.llm_key_edit)

        self.llm_ctx_card = CustomSettingCard(FIF.DICTIONARY, "LLM 上下文长度", "请求参数 num_ctx（0 表示使用模型默认）", self.llm_group)
        self.llm_ctx_spin = DoubleSpinBox()
        self.llm_ctx_spin.setRange(0, 131072)
        self.llm_ctx_spin.setSingleStep(512)
        self.llm_ctx_spin.setDecimals(0)
        self.llm_ctx_card.addWidget(self.llm_ctx_spin)

        self.llm_prompt_card = PromptSettingCard(
            FIF.EDIT, "LLM 提示词", "翻译与润色指令", 110, self.llm_group
        )
        self.llm_prompt_edit = self.llm_prompt_card.editor

        for card in (self.llm_model_card, self.llm_api_card, self.llm_key_card, self.llm_ctx_card, self.llm_prompt_card):
            self.llm_group.addSettingCard(card)
        self.layout.addWidget(self.llm_group)

        self.layout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: transparent; border: none;")


class OverlaySettingInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("overlaySettingInterface")
        self.view = QWidget()
        self.layout = QVBoxLayout(self.view)
        self.layout.setContentsMargins(30, 20, 30, 20)
        self.layout.setSpacing(15)

        self.appear_group = SettingCardGroup("字幕外观", self.view)

        self.show_ocr_card = CustomSettingCard(FIF.LABEL, "同时显示 OCR 原文", "在译文上方显示提取的原始文字", self.appear_group)
        self.sw_show_ocr = SwitchButton()
        self.show_ocr_card.addWidget(self.sw_show_ocr)

        self.ocr_color_card = CustomSettingCard(FIF.PALETTE, "原文颜色", "OCR 原文的文字颜色", self.appear_group)
        self.ocr_color_btn = ColorButton("#FFFF88")
        self.ocr_color_card.addWidget(self.ocr_color_btn)

        self.trans_color_card = CustomSettingCard(FIF.FONT, "译文颜色", "翻译结果的文字颜色", self.appear_group)
        self.trans_color_btn = ColorButton("#FFFFFF")
        self.trans_color_card.addWidget(self.trans_color_btn)

        

        for card in (self.show_ocr_card, self.ocr_color_card, self.trans_color_card):
            self.appear_group.addSettingCard(card)
        self.layout.addWidget(self.appear_group)

        self.overlay_group = SettingCardGroup("贴字设置", self.view)

        self.sw_overlay_card = CustomSettingCard(
            FIF.PIN,
            "启用贴字翻译",
            "在原文位置叠加译文（需 deepseek-ocr 类模型）",
            self.overlay_group
        )
        self.sw_overlay_ocr = SwitchButton()
        self.sw_overlay_card.addWidget(self.sw_overlay_ocr)

        self.overlay_prompt_card = PromptSettingCard(
            FIF.CAMERA,
            "贴字 OCR 提示词",
            "用于带坐标 OCR；去掉提示词中的 \\n 可让模型尽量按行输出",
            90,
            self.overlay_group
        )
        self.overlay_prompt_edit = self.overlay_prompt_card.editor

        self.overlay_min_h_card = CustomSettingCard(
            FIF.BACK_TO_WINDOW,
            "最小贴字文本框高度",
            "贴字模式下每个文本框最小高度 (px)，避免矮字体不可见",
            self.appear_group
        )
        self.overlay_min_h_spin = DoubleSpinBox()
        self.overlay_min_h_spin.setRange(10, 200)
        self.overlay_min_h_spin.setSingleStep(2)
        self.overlay_min_h_card.addWidget(self.overlay_min_h_spin)

        self.sw_debug_box_card = CustomSettingCard(
            FIF.ZOOM,
            "显示 OCR 原始框",
            "贴字模式下额外显示模型返回的检测框（红框）",
            self.overlay_group
        )
        self.sw_overlay_boxes = SwitchButton()
        self.sw_debug_box_card.addWidget(self.sw_overlay_boxes)

        self.sw_auto_merge_card = CustomSettingCard(
            FIF.ALIGNMENT,
            "自动识别换行",
            "开启后，低高度文本行会按规则拼接后再送给下游任务",
            self.overlay_group
        )
        self.sw_auto_merge = SwitchButton()
        self.sw_auto_merge_card.addWidget(self.sw_auto_merge)

        self.min_line_h_card = CustomSettingCard(
            FIF.BACK_TO_WINDOW,
            "最小换行高度",
            "小于该高度的文本框会被视作可拼接行",
            self.overlay_group
        )
        self.min_line_h_spin = DoubleSpinBox()
        self.min_line_h_spin.setRange(1, 300)
        self.min_line_h_spin.setSingleStep(1)
        self.min_line_h_card.addWidget(self.min_line_h_spin)

        self.max_line_gap_card = CustomSettingCard(
            FIF.ALIGNMENT,
            "可拼接框间距（高度）",
            "若上下两个框的间距小于该值 (px)，则可被视作同一句",
            self.overlay_group
        )
        self.max_line_gap_spin = DoubleSpinBox()
        self.max_line_gap_spin.setRange(0, 300)
        self.max_line_gap_spin.setSingleStep(1)
        self.max_line_gap_spin.setDecimals(0)
        self.max_line_gap_card.addWidget(self.max_line_gap_spin)

        self.joiner_card = CustomSettingCard(
            FIF.FONT,
            "拼接字符",
            "用于拼接被视作同一句的文本，英文可用空格，中日文可设为空",
            self.overlay_group
        )
        self.joiner_edit = LineEdit()
        self.joiner_card.addWidget(self.joiner_edit)

        self.line_start_chars_card = CustomSettingCard(
            FIF.FONT,
            "续句起始字符",
            "下一行以这些字符开头时，将判定为上一行续句",
            self.overlay_group
        )
        self.line_start_chars_edit = LineEdit()
        self.line_start_chars_card.addWidget(self.line_start_chars_edit)

        self.line_end_chars_card = CustomSettingCard(
            FIF.FONT,
            "断句结束字符",
            "上一行以这些字符结尾时，将判定为一句结束",
            self.overlay_group
        )
        self.line_end_chars_edit = LineEdit()
        self.line_end_chars_card.addWidget(self.line_end_chars_edit)

        for card in (
            self.sw_overlay_card,
            self.overlay_prompt_card,
            self.sw_debug_box_card,
            self.sw_auto_merge_card,
            self.min_line_h_card,
            self.max_line_gap_card,
            self.joiner_card,
            self.line_start_chars_card,
            self.line_end_chars_card,
            self.overlay_min_h_card,
        ):
            self.overlay_group.addSettingCard(card)

        self.layout.addWidget(self.overlay_group)
        self.layout.addStretch(1)

        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: transparent; border: none;")

        self.sw_auto_merge.checkedChanged.connect(self._sync_merge_controls)
        self._sync_merge_controls(self.sw_auto_merge.isChecked())

    def _sync_merge_controls(self, checked: bool):
        self.min_line_h_card.setVisible(checked)
        self.max_line_gap_card.setVisible(checked)
        self.joiner_card.setVisible(checked)
        self.line_start_chars_card.setVisible(checked)
        self.line_end_chars_card.setVisible(checked)


class AboutInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("aboutInterface")
        self.view = QWidget()
        self.layout = QVBoxLayout(self.view)
        self.layout.setContentsMargins(30, 20, 30, 20)
        self.layout.setSpacing(15)

        self.info_group = SettingCardGroup("关于", self.view)
        self.repo_api = "hanbinhsh/Tsukimi-Translator"
        self.repo_url = "https://github.com/hanbinhsh/Tsukimi-Translator"
        self.author_url = "https://github.com/hanbinhsh"
        self.current_version = "0.1"

        self.version_card = CustomSettingCard(FIF.TAG, "当前版本", f"v{self.current_version}", self.info_group)
        self.author_card = CustomSettingCard(FIF.PEOPLE, "作者", "IceRinne aka. hanbinhsh", self.info_group)
        self.avatar = AvatarLabel(36, self)
        self.avatar.set_url(self.author_url)
        self._load_github_avatar()
        self.author_card.addWidget(self.avatar)

        self.repo_card = CustomSettingCard(
            FIF.LINK,
            "仓库链接",
            "https://github.com/hanbinhsh/Tsukimi-Translator",
            self.info_group,
        )
        self.repo_btn = PushButton(FIF.SHARE, "打开仓库")
        self.repo_btn.clicked.connect(self.open_repo)
        self.repo_card.addWidget(self.repo_btn)

        self.update_card = CustomSettingCard(FIF.UPDATE, "检查更新", "点击检查是否有新版本", self.info_group)
        self.update_btn = PrimaryPushButton("检查更新")
        self.update_btn.clicked.connect(self.check_update)
        self.update_card.addWidget(self.update_btn)

        for card in (
            self.author_card,
            self.version_card,
            self.repo_card,
            self.update_card,
        ):
            self.info_group.addSettingCard(card)

        self.layout.addWidget(self.info_group)
        self.layout.addStretch(1)

        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: transparent; border: none;")

    def open_repo(self):
        QDesktopServices.openUrl(QUrl(self.repo_url))

    def _load_github_avatar(self):
        try:
            api = f"https://api.github.com/users/{self.repo_api.split('/')[0]}"
            resp = requests.get(api, timeout=8)
            resp.raise_for_status()
            avatar_url = resp.json().get("avatar_url", "")
            if avatar_url:
                img = requests.get(avatar_url, timeout=8)
                img.raise_for_status()
                self.avatar.set_avatar_from_bytes(img.content)
        except Exception:
            pass

    def check_update(self):
        try:
            url = f"https://api.github.com/repos/{self.repo_api}/releases/latest"
            resp = requests.get(url, timeout=8)
            changelog = ""
            release_url = self.repo_url
            if resp.status_code == 404:
                # 没有 release 时，降级读取最新 tag
                tag_resp = requests.get(f"https://api.github.com/repos/{self.repo_api}/tags", timeout=8)
                tag_resp.raise_for_status()
                tags = tag_resp.json()
                latest = tags[0]["name"] if tags else "unknown"
                release_url = f"{self.repo_url}/releases"
                changelog = "该版本暂无 Release 日志（仓库可能仅使用 Tags 发布）。"
            else:
                resp.raise_for_status()
                data = resp.json()
                latest = data.get("tag_name") or data.get("name") or "unknown"
                release_url = data.get("html_url") or release_url
                changelog = (data.get("body") or "").strip() or "该版本未提供更新日志。"

            if latest.lstrip("vV") == self.current_version.lstrip("vV"):
                InfoBar.success("检查更新", f"当前已是最新版本：{self.current_version}", parent=self)
            else:
                content = (
                    f"当前版本：{self.current_version}\n"
                    f"最新版本：{latest}\n\n"
                    f"更新日志：\n{changelog}\n\n"
                    "是否前往更新页面？"
                )
                box = MessageBox("发现新版本", content, self)
                box.yesButton.setText("立即更新")
                box.cancelButton.setText("稍后")
                box.setClosableOnMaskClicked(True)
                box.setDraggable(True)
                if box.exec():
                    QDesktopServices.openUrl(QUrl(release_url))
        except Exception as e:
            InfoBar.error("检查更新失败", f"无法连接 GitHub API：{e}", parent=self)


# ══════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════

class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.app_icon = load_app_icon()
        setTheme(Theme.DARK)
        self.setWindowTitle("Tsukimi Translator")
        self.setWindowIcon(self.app_icon)
        self.resize(820, 820)

        self.home_page    = HomeInterface(self)
        self.setting_page = SettingInterface(self)
        self.ai_page      = AISettingInterface(self)
        self.overlay_page = OverlaySettingInterface(self)
        self.about_page   = AboutInterface(self)

        self.addSubInterface(self.home_page,    FIF.HOME,    "主页")
        self.addSubInterface(self.setting_page, FIF.SETTING, "配置")
        self.addSubInterface(self.ai_page,      FIF.ROBOT,   "AI 配置")
        self.addSubInterface(self.overlay_page, FIF.BRUSH,   "贴字设置")
        self.addSubInterface(
            self.about_page,
            FIF.INFO,
            "关于",
            NavigationItemPosition.BOTTOM,
        )

        # 右侧页面固定底部操作条
        self._content_host = self.stackedWidget if hasattr(self, "stackedWidget") else self
        self.top_action_bar = QFrame(self._content_host)
        self.top_action_bar.setObjectName("topActionBar")
        self.top_action_bar.setStyleSheet(
            "#topActionBar {"
            "background: transparent;"
            "border-top: 1px solid rgba(120,120,120,0.35);"
            "}"
        )
        self.top_action_layout = QHBoxLayout(self.top_action_bar)
        self.top_action_layout.setContentsMargins(16, 8, 16, 8)
        self.top_action_layout.setSpacing(8)
        self.top_action_layout.addStretch(1)

        self.start_nav_btn = PrimaryPushButton("启动翻译", self.top_action_bar)
        self.save_nav_btn = PushButton("保存设置", self.top_action_bar)
        self.reset_nav_btn = PushButton("重置设置", self.top_action_bar)
        self.start_nav_btn.setFixedWidth(140)
        self.save_nav_btn.setFixedWidth(140)
        self.reset_nav_btn.setFixedWidth(110)
        self.top_action_layout.addWidget(self.reset_nav_btn)
        self.top_action_layout.addWidget(self.save_nav_btn)
        self.top_action_layout.addWidget(self.start_nav_btn)

        self.start_nav_btn.clicked.connect(self.toggle_overlay)
        self.save_nav_btn.clicked.connect(self.save_all)
        self.reset_nav_btn.clicked.connect(self.reset_all_settings)

        # 给右侧各页面预留底部固定栏空间，滚动内容在其上方结束
        for page in (self.home_page, self.setting_page, self.ai_page, self.overlay_page, self.about_page):
            page.setViewportMargins(0, 0, 0, 52)

        self.load_settings()
        self._sync_region_ui()

        # 恢复上次选择的屏幕
        saved_name = self.cfg.get("capture_screen_name", "")
        for i in range(self.home_page.screen_combo.count()):
            if self.home_page.screen_combo.itemData(i) == saved_name:
                self.home_page.screen_combo.setCurrentIndex(i)
                break

        # 信号绑定
        self.home_page.refresh_btn.clicked.connect(self.refresh_windows)
        self.home_page.region_btn.clicked.connect(self.open_region_selector)
        self.home_page.clear_region_btn.clicked.connect(self.clear_region)
        self.stackedWidget.currentChanged.connect(self._on_page_changed)

        self.refresh_windows()
        self.overlay = None
        self._layout_top_action_bar()
        self._refresh_start_button_style()
        self._on_page_changed(self.stackedWidget.currentIndex())

    def _layout_top_action_bar(self):
        bar_h = 52
        self.top_action_bar.setGeometry(0, self._content_host.height() - bar_h, self._content_host.width(), bar_h)
        self.top_action_bar.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_top_action_bar()

    def _refresh_start_button_style(self):
        running = self.overlay is not None
        if running:
            self.start_nav_btn.setText("停止翻译")
            self.start_nav_btn.setStyleSheet(
                "QPushButton{background:#b42318;color:white;border-radius:6px;padding:6px 12px;}"
                "QPushButton:hover{background:#cf3024;}"
            )
        else:
            self.start_nav_btn.setText("启动翻译")
            self.start_nav_btn.setStyleSheet(
                "QPushButton{background:#0e9f6e;color:white;border-radius:6px;padding:6px 12px;}"
                "QPushButton:hover{background:#13b87f;}"
            )

    def _on_page_changed(self, index: int):
        page = self.stackedWidget.widget(index)
        show_save = page in (self.setting_page, self.ai_page, self.overlay_page)
        self.save_nav_btn.setVisible(show_save)
        self.reset_nav_btn.setVisible(show_save)

    def reset_all_settings(self):
        box = MessageBox(
            "确认重置所有设置？",
            "该操作会立即用默认值覆盖当前配置，且无法撤销。",
            self
        )
        box.setClosableOnMaskClicked(True)
        box.setDraggable(True)
        if not box.exec():
            return
        from config_manager import DEFAULT_CONFIG
        self.cfg = DEFAULT_CONFIG.copy()
        save_config(self.cfg)
        self.load_settings()
        if self.overlay:
            self.overlay.cfg = self.cfg
            self.overlay.update_window_flags()
            self.overlay.apply_mode()
            self.overlay.update_layout_settings()
        InfoBar.success("已重置", "所有设置已恢复默认值", parent=self)

    # ── 设置加载 ──

    def load_settings(self):
        s = self.setting_page
        mode = self.cfg.get("capture_mode", "interval")
        if mode not in ("interval", "trigger"): mode = "interval"
        s.mode_seg.setCurrentItem(mode)
        s.on_mode_changed(mode)

        # 截图来源
        source = self.cfg.get("capture_source", "window")
        if source not in ("window", "region"): source = "window"
        self.home_page.source_seg.setCurrentItem(source)
        self.home_page.on_source_changed(source)

        grow = self.cfg.get("grow_direction", "up")
        if grow not in ("up", "down"): grow = "up"
        s.grow_seg.setCurrentItem(grow)

        s.width_spin.setValue(self.cfg.get("ui_max_width", 800))
        s.sw_hide.setChecked(self.cfg.get("auto_hide", True))
        s.interval_spin.setValue(self.cfg.get("capture_interval", 2.5))
        s.trigger_combo.setCurrentText(self.cfg.get("trigger_key", "Left Click"))
        s.sw_visible.setChecked(self.cfg.get("window_visible", True))
        s.sw_top.setChecked(self.cfg.get("always_on_top", True))

        self.ai_page.ocr_model_edit.setText(self.cfg.get("ocr_model", ""))
        self.ai_page.ocr_api_edit.setText(self.cfg.get("ocr_api", "http://localhost:11434/api/generate"))
        self.ai_page.ocr_key_edit.setText(self.cfg.get("ocr_key", ""))
        self.ai_page.ocr_ctx_spin.setValue(self.cfg.get("ocr_context_length", 8192))
        self.ai_page.ocr_prompt_edit.setText(self.cfg.get("ocr_prompt", ""))
        self.ai_page.llm_model_edit.setText(self.cfg.get("llm_model", ""))
        self.ai_page.llm_api_edit.setText(self.cfg.get("llm_api", "http://localhost:11434/api/generate"))
        self.ai_page.llm_key_edit.setText(self.cfg.get("llm_key", ""))
        self.ai_page.llm_ctx_spin.setValue(self.cfg.get("llm_context_length", 8192))
        self.ai_page.llm_prompt_edit.setText(self.cfg.get("llm_prompt", ""))
        s.scale_spin.setValue(self.cfg.get("scale_factor", 0.5))
        s.sw_stream.setChecked(self.cfg.get("use_stream", False))
        s.sw_ocr.setChecked(self.cfg.get("use_ocr", True))
        s.sw_llm.setChecked(self.cfg.get("use_llm", True))
        s.sw_copy.setChecked(self.cfg.get("auto_copy", False))
        s.sw_remove_blank.setChecked(self.cfg.get("remove_blank_lines", False))

        self.overlay_page.sw_show_ocr.setChecked(self.cfg.get("show_ocr_text", False))
        self.overlay_page.ocr_color_btn.setColor(self.cfg.get("ocr_color", "#FFFF88"))
        self.overlay_page.trans_color_btn.setColor(self.cfg.get("trans_color", "#FFFFFF"))
        self.overlay_page.overlay_min_h_spin.setValue(self.cfg.get("overlay_min_box_height", 28))
        self.overlay_page.sw_overlay_ocr.setChecked(self.cfg.get("use_overlay_ocr", False))
        self.overlay_page.overlay_prompt_edit.setText(
            self.cfg.get("overlay_ocr_prompt", "\n<|grounding|>OCR the image.")
        )
        self.overlay_page.sw_overlay_boxes.setChecked(
            self.cfg.get("show_overlay_debug_boxes", False)
        )
        self.overlay_page.sw_auto_merge.setChecked(
            self.cfg.get("overlay_auto_merge_lines", False)
        )
        self.overlay_page.min_line_h_spin.setValue(
            self.cfg.get("overlay_min_line_height", 40)
        )
        self.overlay_page.max_line_gap_spin.setValue(
            self.cfg.get("overlay_max_line_gap", 4)
        )
        self.overlay_page.joiner_edit.setText(
            self.cfg.get("overlay_joiner", " ")
        )
        self.overlay_page.line_start_chars_edit.setText(
            self.cfg.get("line_start_chars", ",.;:!?)]}、，。！？；：」』）】》")
        )
        self.overlay_page.line_end_chars_edit.setText(
            self.cfg.get("line_end_chars", ".!?。！？…")
        )

    def _sync_region_ui(self):
        region = self.cfg.get("capture_region")
        screen_name = self.cfg.get("capture_screen_name", "")
        if region:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            self.home_page.region_card.subLabel.setText(
                f"已框选 [{screen_name}]  偏移({x}, {y})  {w}×{h} 逻辑px"
            )
            self.home_page.clear_region_btn.setEnabled(True)
        else:
            self.home_page.region_card.subLabel.setText("全窗口（未框选）")
            self.home_page.clear_region_btn.setEnabled(False)

    # ── 设置保存 ──

    def save_all(self):
        s = self.setting_page
        self.cfg.update({
            "capture_source":      get_seg_key(self.home_page.source_seg, "window"),
            "capture_mode":        get_seg_key(s.mode_seg, "interval"),
            "grow_direction":      get_seg_key(s.grow_seg, "up"),
            "capture_interval":    s.interval_spin.value(),
            "trigger_key":         s.trigger_combo.currentText(),
            "ui_max_width":        int(s.width_spin.value()),
            "auto_hide":           s.sw_hide.isChecked(),
            "window_visible":      s.sw_visible.isChecked(),
            "always_on_top":       s.sw_top.isChecked(),
            "show_ocr_text":       self.overlay_page.sw_show_ocr.isChecked(),
            "ocr_color":           self.overlay_page.ocr_color_btn.color(),
            "trans_color":         self.overlay_page.trans_color_btn.color(),
            "overlay_min_box_height": int(self.overlay_page.overlay_min_h_spin.value()),
            "ocr_model":           self.ai_page.ocr_model_edit.text(),
            "ocr_api":             self.ai_page.ocr_api_edit.text(),
            "ocr_key":             self.ai_page.ocr_key_edit.text(),
            "ocr_context_length":  int(self.ai_page.ocr_ctx_spin.value()),
            "ocr_prompt":          self.ai_page.ocr_prompt_edit.toPlainText(),
            "llm_model":           self.ai_page.llm_model_edit.text(),
            "llm_api":             self.ai_page.llm_api_edit.text(),
            "llm_key":             self.ai_page.llm_key_edit.text(),
            "llm_context_length":  int(self.ai_page.llm_ctx_spin.value()),
            "scale_factor":        s.scale_spin.value(),
            "use_stream":          s.sw_stream.isChecked(),
            "use_ocr":             s.sw_ocr.isChecked(),
            "use_llm":             s.sw_llm.isChecked(),
            "auto_copy":           s.sw_copy.isChecked(),
            "remove_blank_lines":  s.sw_remove_blank.isChecked(),
            "use_overlay_ocr":     self.overlay_page.sw_overlay_ocr.isChecked(),
            "show_overlay_debug_boxes": self.overlay_page.sw_overlay_boxes.isChecked(),
            "overlay_auto_merge_lines": self.overlay_page.sw_auto_merge.isChecked(),
            "overlay_ocr_prompt": self.overlay_page.overlay_prompt_edit.toPlainText(),
            "overlay_min_line_height": int(self.overlay_page.min_line_h_spin.value()),
            "overlay_max_line_gap": int(self.overlay_page.max_line_gap_spin.value()),
            "overlay_joiner": self.overlay_page.joiner_edit.text(),
            "line_start_chars": self.overlay_page.line_start_chars_edit.text(),
            "line_end_chars": self.overlay_page.line_end_chars_edit.text(),
            "llm_prompt":          self.ai_page.llm_prompt_edit.toPlainText(),
            "target_hwnd":         self.home_page.combo.currentData(),
            "capture_screen_name": self.home_page.screen_combo.currentData() or "",
        })
        save_config(self.cfg)

        if self.overlay:
            self.overlay.cfg = self.cfg
            self.overlay.update_window_flags()
            self.overlay.apply_mode()
            self.overlay.update_layout_settings()
            # 贴字模式关闭时，销毁浮层
            if not self.cfg.get("use_overlay_ocr", False):
                if self.overlay.text_overlay and isValid(self.overlay.text_overlay):
                    self.overlay.text_overlay.close()
                    self.overlay.text_overlay = None

        InfoBar.success("保存成功", "所有配置已生效", parent=self)

    # ── 窗口刷新 ──

    def refresh_windows(self):
        self.home_page.combo.clear()
        for hwnd, title in get_active_windows():
            self.home_page.combo.addItem(title, userData=hwnd)

    # ── 翻译启停 ──

    def toggle_overlay(self):
        if not self.overlay:
            source = get_seg_key(self.home_page.source_seg, "window")
            target = self.home_page.combo.currentData()
            region = self.cfg.get("capture_region")

            if source == "window" and not target:
                InfoBar.warning("提示", "请先选择游戏窗口", parent=self)
                return
            if source == "region" and not region:
                InfoBar.warning("提示", "请先在目标屏幕上框选截图区域", parent=self)
                return

            self.cfg["capture_source"]      = source
            self.cfg["target_hwnd"]         = target if source == "window" else 0
            self.cfg["capture_screen_name"] = self.home_page.screen_combo.currentData() or ""
            self.overlay = SubtitleOverlay(self.cfg)
            if self.cfg.get("window_visible", True):
                self.overlay.show()
            self._refresh_start_button_style()
        else:
            self.overlay.close()
            self.overlay = None
            self._refresh_start_button_style()

    # ── 区域框选 ──

    def open_region_selector(self):
        """最小化主窗口，隐藏浮窗，延迟弹出框选器"""
        self.showMinimized()
        if self.overlay:
            self._overlay_was_visible = self.overlay.isVisible()
            self.overlay.hide()
        QTimer.singleShot(350, self._launch_selector)

    def _launch_selector(self):
        screen = self.home_page.selected_screen()
        self._selector = RegionSelector(screen)
        self._selector.region_selected.connect(
            lambda x, y, w, h: self.on_region_selected(x, y, w, h, screen.name())
        )
        self._selector.closed.connect(self._on_selector_closed)

    def _on_selector_closed(self):
        self.showNormal()
        if self.overlay and getattr(self, "_overlay_was_visible", True):
            self.overlay.show()

    def on_region_selected(self, x, y, w, h, screen_name):
        self.cfg["capture_region"]      = {"x": x, "y": y, "w": w, "h": h}
        self.cfg["capture_screen_name"] = screen_name
        save_config(self.cfg)
        self._sync_region_ui()
        if self.overlay:
            self.overlay.cfg = self.cfg

    def clear_region(self):
        self.cfg["capture_region"] = None
        save_config(self.cfg)
        self._sync_region_ui()
        if self.overlay:
            self.overlay.cfg = self.cfg


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
