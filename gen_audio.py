from gradio_client import Client, handle_file
import os
import shutil

client = Client("http://xxx:50000/")

result = client.predict(
  tts_text="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
  mode_checkbox_group="3s极速复刻",
  sft_dropdown=None,
  prompt_text="卧槽，他妈的合合分支的时候被人覆盖了吧卧槽，你信不信",
  prompt_wav_upload=handle_file('小王.m4a'),
  prompt_wav_record=None,
  instruct_text="用粤语说这句话",
  seed=0,
  stream="false",
  speed=1,
  api_name="/generate_audio"
)
print(result)

# 指定您希望保存文件的目录
custom_save_dir = 'tmp/'

# 确保保存目录存在
os.makedirs(custom_save_dir, exist_ok=True)

# 获取文件名
file_name = os.path.basename(result)

# 构建新的文件路径
new_file_path = os.path.join(custom_save_dir, file_name)

# 将文件移动到新的目录
shutil.move(result, new_file_path)

# 打印新的文件路径
print(f"File saved to: {new_file_path}")
