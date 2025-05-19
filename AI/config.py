# some config about the AI

import requests
import json

# 1. 配置API基本信息
API_URL = "https://api.example.com/endpoint "  # 替换为实际API地址
API_KEY = "your_api_key_here"                
HEADERS = {
    "Content-Type": "application/json",       # 避免出现 application/octet-stream 错误 
    "Authorization": f"Bearer {API_KEY}"      # 根据API文档选择合适的认证方式 
}

# 2. 定义请求参数（根据API文档调整）
PAYLOAD = {
    "task": "example_task",
    "parameters": {
        "input": "test_data",
        "option": True
    }
}

# 3. 发送API请求（支持GET/POST，根据API需求修改）
try:
    response = requests.post(                 # 或 requests.get() 
        API_URL,
        headers=HEADERS,
        json=PAYLOAD,                          # 自动序列化为JSON并设置Content-Type 
        timeout=10                             # 设置超时时间
    )
    
    # 4. 处理响应
    if response.status_code == 200:
        result = response.json()              # 解析JSON响应 
        print("任务执行成功:", json.dumps(result, indent=2))
    else:
        print(f"请求失败 (状态码 {response.status_code}):")
        print(response.text)

except requests.exceptions.RequestException as e:
    print("网络请求异常:", str(e))

