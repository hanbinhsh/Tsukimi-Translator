import win32gui
import ctypes
from ctypes import wintypes

# 常量定义
DWMWA_EXTENDED_FRAME_BOUNDS = 9
def get_active_windows():
    """获取所有可见且有标题的窗口句柄和标题"""
    window_list = []
    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                window_list.append((hwnd, title))
    win32gui.EnumWindows(enum_handler, None)
    return window_list

def get_window_rect(hwnd):
    """获取窗口在屏幕上的真实视觉坐标 (排除隐形边框)"""
    
    # 1. 强制告诉 Windows，本程序支持高 DPI
    # 防止因为 125% 或 150% 缩放导致的坐标偏移
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1) # 1 = Process_System_DPI_Aware
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    # 2. 使用 DWM API 获取真实的窗口边界 (排除阴影)
    rect = wintypes.RECT()
    DWM = ctypes.windll.dwmapi
    DWM.DwmGetWindowAttribute(
        hwnd, 
        DWMWA_EXTENDED_FRAME_BOUNDS, 
        ctypes.byref(rect), 
        ctypes.sizeof(rect)
    )
    
    # 如果 DWM 获取失败（比如某些特殊窗口），回退到旧的 GetWindowRect
    if rect.left == 0 and rect.top == 0 and rect.right == 0:
        import win32gui
        rect_tuple = win32gui.GetWindowRect(hwnd)
        return rect_tuple[0], rect_tuple[1], rect_tuple[2]-rect_tuple[0], rect_tuple[3]-rect_tuple[1]

    x = rect.left
    y = rect.top
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    
    return x, y, w, h