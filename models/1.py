from openai import OpenAI

# ---------------- 配置区域 ----------------
# 因为你使用了 SSH 反向隧道 (ssh -R 8079:localhost:8079 ...)
# 远程服务器的 8079 端口已经被直接连通到了你本地 WSL 的 8079 端口
API_BASE_URL = "http://localhost:8079/v1"

# 这里的路径必须和你本地 WSL 启动 vLLM 时 --model 后面的名称/路径一模一样！
# 如果你在 WSL 里启动用的是缩写比如 model="Qwen/Qwen3.6-27B"，这里也要保持一致
MODEL_NAME = "/home/yunlei/model/Qwen3.6-27B-FP8" 
# ------------------------------------------

# 初始化客户端
client = OpenAI(
    base_url=API_BASE_URL,
    api_key="EMPTY"  # vLLM 本地调用不需要鉴权
)

def main():
    print(f"🚀 准备连接通过 SSH 隧道映射的本地服务: {API_BASE_URL}")
    
    # 准备测试问题
    messages = [
        {"role": "system", "content": "你是一个严谨且幽默的 AI 助手。"},
        {"role": "user", "content": "你好，Qwen！请确认你是否在线，并倒着拼写出 'WSL Ubuntu'。"}
    ]

    try:
        print("⏳ 正在发送请求，请观察你本地 WSL 终端是否有日志刷新...\n")
        
        # 发起流式/非流式调用
        chat_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )

        # 打印成功结果
        print("✅ 调用成功！")
        print("🤖 Qwen3.6 的回复：")
        print("-" * 50)
        print(chat_response.choices[0].message.content)
        print("-" * 50)

    except ConnectionError:
        print("❌ 连接被拒绝！")
        print("请检查：\n1. 你本地的 WSL vLLM 服务是否还在正常运行？\n2. 你的 SSH 反向隧道 (ssh -R) 终端窗口是否不小心关掉了？")
    except Exception as e:
        print(f"❌ 发生未知错误：\n{e}")

if __name__ == "__main__":
    main()