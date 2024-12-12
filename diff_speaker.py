import datetime
import numpy as np
# 通过比较 embedding 确认说话人身份
def identify_speaker(new_emb, known_embeddings, threshold=0.34):
    for speaker, emb in known_embeddings.items():
        # 计算两个embedding之间的余弦相似度
        sim = np.dot(new_emb, emb) / (np.linalg.norm(new_emb) * np.linalg.norm(emb))
        if sim >= threshold:
            return speaker

    # 获取当前时间戳并格式化为字符串
    cts = datetime.datetime.now().timestamp()
    # 获取整数秒
    cts_s = int(cts)
    return f"unknown_{cts_s}"



import glob
import os
# 保存embedding的目录
voice_npy_dir = 'voice_npy/'
# 使用glob找到所有的.npy文件
npy_files = glob.glob(os.path.join(voice_npy_dir, '*.npy'))
# 加载已知的说话人
known_speakers = {}
# 遍历文件列表并加载每个.npy文件
for file_path in npy_files:
    # 提取文件名（不包含路径和扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # 加载embedding
    known_speakers[file_name] = np.load(file_path)



from modelscope.pipelines import pipeline
sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

zhikai1 = 'zhikai1_16k.wav'
zhikai2 = 'zhikai2_16k.wav'
luoxin1 = 'luoxin1_16k.wav'

# 相同说话人语音
# result = sv_pipeline([zhikai1, zhikai2])
# print(result['text']=='yes')

# 不同说话人语音
# result = sv_pipeline([zhikai1, luoxin1])
# print(result['text']=='yes')

# 可以自定义得分阈值来进行识别，阈值越高，判定为同一人的条件越严格
# result = sv_pipeline([zhikai1, luoxin1], thr=0.34)
# print(result)

# 可以传入output_emb参数，输出结果中就会包含提取到的说话人embedding
# result = sv_pipeline([zhikai1, luoxin1], output_emb=True)
# print(result['embs'], result['outputs'])



xianbo1 = 'luoxin2_16k.wav'
# 提取新的人声embedding
new_voice_result = sv_pipeline([xianbo1], output_emb=True)
new_voice_emb = new_voice_result['embs'][0]

identified_speaker = identify_speaker(new_voice_emb, known_speakers, threshold=0.6)
print(identified_speaker)
if identified_speaker.startswith("unknown_"):
    # 可以传入save_dir参数，提取到的说话人embedding会存储在save_dir目录中
    result = sv_pipeline([xianbo1], save_dir=voice_npy_dir) # 保存的文件名与读取的文件名一致

