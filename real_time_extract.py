import pyaudio
import wave
import threading
from queue import Queue
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 录音帧
frames = []

# 使用队列来存储音频数据
audio_queue = Queue()

# 完整的记录
full_text = ""

# 录音参数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1              # 声道数
RATE = 44100              # 采样率
CHUNK = 1024              # 数据块大小
RECORD_SECONDS = 1        # 录音时间
WAVE_OUTPUT_FILENAME = "output.wav"  # 输出文件名
WAVE_OUTPUT_FILENAME_TMP = "output_tmp.wav"

# 录制音频线程
class RecordAudio:
    def __init__(self):
        self._running = True

    def terminate(self, audio_queue):
        self._running = False
        audio_queue.put(None)
        self._thread.join()

    def run(self, stream, frames):
        self._thread = threading.current_thread()
        while self._running:
            tframes = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK)
                    frames.append(data)
                    tframes.append(data)
                except Exception as e:
                    print(f"录音过程中发生错误: {e}")
                    self.terminate(audio_queue)

            audio_queue.put(tframes)

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    disable_pbar=True,
)

# 识别音频线程
class ExtractText():
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False
        self._thread.join()

    def run(self):
        self._thread = threading.current_thread()
        while self._running:
            if audio_queue.empty():
                continue
            # 从队列中获取音频数据
            data = audio_queue.get()
            if data is None:
                break  # 如果接收到None，则结束线程
            try:
                # 保存音频
                tf = wave.open(WAVE_OUTPUT_FILENAME_TMP, 'wb')
                tf.setnchannels(CHANNELS)
                tf.setsampwidth(p.get_sample_size(FORMAT))
                tf.setframerate(RATE)
                tf.writeframes(b''.join(data))
                tf.close()

                # 尝试识别音频
                input=f"%s"%WAVE_OUTPUT_FILENAME_TMP
                res = model.generate(
                    input=input,
                    cache={},
                    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=False, # 输出结果中是否包含标点与逆文本正则化。
                    batch_size_s=60, # 表示采用动态batch，batch中总音频时长，单位为秒s。
                    vad_model="fsmn-vad", # 表示开启VAD，VAD的作用是将长音频切割成短音频，此时推理耗时包括了VAD与SenseVoice总耗时，为链路耗时，如果需要单独测试SenseVoice模型耗时，可以关闭VAD模型。
                    merge_vad=True,  # 是否将 vad 模型切割的短音频碎片合成，合并后长度为merge_length_s，单位为秒s。
                    merge_length_s=15, # 是否将 vad 模型切割的短音频碎片合成，合并后长度为merge_length_s，单位为秒s。
                    ban_emo_unk=True, # 禁用emo_unk标签，禁用后所有的句子都会被赋与情感标签。默认False
                )
                text = rich_transcription_postprocess(res[0]["text"])
                if len(text) > 0 :
                    global full_text
                    full_text += text
                    print(f"识别结果: {text}")
            except Exception as e :
                print(f"无法识别结果; {e}")

# 初始化pyaudio
p = pyaudio.PyAudio()

# 打开流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# 启动音频识别线程
extract_text_task = ExtractText()
extract_text_th = threading.Thread(target=extract_text_task.run, args=())
extract_text_th.start()

# 启动录制音频线程
record_audio_task = RecordAudio()
record_audio_th = threading.Thread(target=record_audio_task.run, args=(stream,frames))
record_audio_th.start()


print("等待用户输入来停止录音")
# 等待用户输入来停止录音
input()

print("停止录音")
# 停止录音线程
record_audio_task.terminate(audio_queue)

print("录音结束")

# 停止流
stream.stop_stream()
stream.close()
# 关闭pyaudio
p.terminate()

# 保存完整录音
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print("录音保存结束")
print(full_text)
