import os
import requests
import wave
from requests.exceptions import ChunkedEncodingError


def call_endpoint(endpoint, data, files, output_filename):
    """
    调用指定接口，并将返回的 PCM 数据保存为带 WAV 头的音频文件。
    如果流式读取出现异常，则改用 response.content 直接获取数据。
    """
    print(f"正在调用 {endpoint} ...")
    response = requests.post(endpoint, data=data, files=files, stream=True)
    response.raise_for_status()

    pcm_data = b""
    try:
        # 尝试流式读取响应内容
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                pcm_data += chunk
    except ChunkedEncodingError as e:
        print("捕获到 ChunkedEncodingError，直接使用 response.content。")
        pcm_data = response.content

    # 保存为 WAV 文件，设置单声道、16 位采样、采样率 16000 Hz
    output_path = os.path.join("output", output_filename)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16 位采样 (2 字节)
        wf.setframerate(16000)  # 采样率
        wf.writeframes(pcm_data)
    print(f"音频已保存为 {output_path}")


def main():
    # 确保 output 文件夹存在
    os.makedirs("output", exist_ok=True)

    base_url = "http://localhost:50000"

    # --------------------------
    # 1. 调用 /inference_sft 接口
    # 参数: tts_text, spk_id
    endpoint_sft = f"{base_url}/inference_sft"
    data_sft = {
        "tts_text": "你好，我是测试文本，用于测试 sft 接口。",
        "spk_id": "中文女"
    }
    # 此接口不需要上传文件，files传入空字典即可
    call_endpoint(endpoint_sft, data=data_sft, files={}, output_filename="inference_sft.wav")

    # --------------------------
    # 2. 调用 /inference_zero_shot 接口
    # 参数: tts_text, prompt_text, prompt_wav
    endpoint_zero_shot = f"{base_url}/inference_zero_shot"
    data_zero_shot = {
        "tts_text": "这是一段测试文本，用于测试 zero_shot 接口。",
        "prompt_text": "希望你以后能够做的比我还好呦。"
    }
    # 每次上传文件时重新打开文件
    with open("myvoice_news_en_out.wav", "rb") as f:
        files_zero_shot = {"prompt_wav": f}
        call_endpoint(endpoint_zero_shot, data=data_zero_shot, files=files_zero_shot, output_filename="inference_zero_shot.wav")

    # --------------------------
    # 3. 调用 /inference_cross_lingual 接口
    # 参数: tts_text, prompt_wav
    endpoint_cross_lingual = f"{base_url}/inference_cross_lingual"
    data_cross_lingual = {
        "tts_text": "在神秘的永晓森林里，一个名叫利奥的好奇男孩在一棵古老的柳树后发现了一个闪闪发光的隐藏传送门。"
    }
    with open("myvoice_news_en_out.wav", "rb") as f:
        files_cross_lingual = {"prompt_wav": f}
        call_endpoint(endpoint_cross_lingual, data=data_cross_lingual, files=files_cross_lingual, output_filename="inference_cross_lingual.wav")

    # --------------------------
    # 4. 调用 /inference_instruct 接口
    # 参数: tts_text, spk_id, instruct_text
    endpoint_instruct = f"{base_url}/inference_instruct"
    data_instruct = {
        "tts_text": "在面对挑战时，他展现了非凡的勇气与智慧。",
        "spk_id": "中文男",
        "instruct_text": "请用更富有感情的语气朗读这段话。"
    }
    call_endpoint(endpoint_instruct, data=data_instruct, files={}, output_filename="inference_instruct.wav")

    # --------------------------
    # 5. 调用 /inference_instruct2 接口
    # 参数: tts_text, instruct_text, prompt_wav
    endpoint_instruct2 = f"{base_url}/inference_instruct2"
    data_instruct2 = {
        "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        "instruct_text": "用四川话说这句话。"
    }
    with open("myvoice_news_en_out.wav", "rb") as f:
        files_instruct2 = {"prompt_wav": f}
        call_endpoint(endpoint_instruct2, data=data_instruct2, files=files_instruct2, output_filename="inference_instruct2.wav")


if __name__ == "__main__":
    main()
