import requests
import base64
import json
import os

def test_aggregated_ocr(image_path, model_name="deepseek-ocr"):
    # 1. 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 {image_path}")
        return

    # 2. 将图片读取并转换为 Base64 编码
    print(f"正在读取并编码图片: {image_path}...")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # 3. 构造核心 Prompt（复现场景 B 的关键）
    # 使用 '\\n' 在 Python 中表示发送一个字面的反斜杠 '\' 和字母 'n'
    # 这完全模拟了你在 PowerShell 中双引号直接传递 "\n" 而未被解析为换行符的畸形输入
    prompt_text = "\\n<|grounding|>OCR the image."

    # 4. 配置 API 请求数据
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "images": [base64_image],
        "stream": False,  # 关闭流式输出，直接获取完整结果方便对比
        "options": {
            "temperature": 0.0, # OCR 任务建议设置为 0，保证输出的稳定性
        }
    }

    # 5. 发送请求
    print(f"正在调用 Ollama API ({url})，等待模型推理...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # 检查 HTTP 状态码
        
        # 6. 解析并打印结果
        result_data = response.json()
        print("\n" + "="*40)
        print("🎯 模型返回的聚合格式结果 (Scenario B):")
        print("="*40)
        print(result_data.get("response", ""))
        print("="*40)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ API 请求失败: {e}")
        if response is not None:
            print(f"返回的详细信息: {response.text}")

if __name__ == "__main__":
    # 使用你之前的测试图片路径
    target_image = r"G:\QtProjects\Translator\debug_current_vision.png"
    
    # 运行测试
    test_aggregated_ocr(target_image)