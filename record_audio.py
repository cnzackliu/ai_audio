import pyaudio
import wave
import threading

# 录音参数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1              # 声道数
RATE = 44100              # 采样率
CHUNK = 1024              # 数据块大小
RECORD_SECONDS = 2        # 录音时间
WAVE_OUTPUT_FILENAME = "output.wav"  # 输出文件名
WAVE_OUTPUT_FILENAME_TMP = "output_tmp.wav"

class AudioRecorder:
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False
        self._thread.join()

    def run(self, stream, frames):
        self._thread = threading.current_thread()
        while self._running:
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK)
                    frames.append(data)
                except Exception as e:
                    print(f"录音过程中发生错误: {e}")
                    self.terminate()

# 初始化pyaudio
p = pyaudio.PyAudio()

# 打开流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("按下回车键停止录音...")

# 开始录音
frames = []

task = AudioRecorder()
th = threading.Thread(target=task.run, args=(stream, frames,))
th.start()

print("等待用户输入来停止录音")
# 等待用户输入来停止录音
input()

print("停止录音")
# 停止录音线程
task.terminate()

print("录音结束")

# 停止流
stream.stop_stream()
stream.close()
# 关闭pyaudio
p.terminate()

# 保存录音
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print("录音保存结束")
