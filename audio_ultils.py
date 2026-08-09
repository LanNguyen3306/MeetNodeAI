'''
Các hàm xử lí audio: ắt file dài thành nhiều đoạn nhỏ để model
AI xử lí hiệu quả hơn và tránh tràn bộ nhớ với file quá lớn
'''

import math
import os
from pydub import AudioSegment

def split_audio(audio_path:str, temp_dir: str, segment_minutes: int = 10)-> list[str]:
    '''
    Cắt file audio thành nhiều đoạn nhỏ
    :param audio_path:
    :param temp_dir:
    :param segment_minutes:
    :return: Danh sách đường dẫn tới ca file đoạn nhỏ theo đúng thứ tự thời gian

    '''
    os.makedirs(temp_dir, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)

    segment_ms = segment_minutes * 60 * 1000
    total_ms = len(audio)
    num_segments = math.ceil(total_ms / segment_ms) if total_ms > 0 else 1

    segment_paths = []
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for i in range(num_segments):
        start = i * segment_ms
        end = min((i+1) * segment_ms, total_ms)
        chunk = audio[start:end]

        chunk_path = os.path.join(temp_dir, f"{base_name}_path{i:03d}.wav")
        # Xuat ra wav 16kHz moni - Dinh dang whisper xu li nhanh va on dinh nhat
        chunk = chunk.set_frame_rate(16000).set_channels(1)
        chunk.export(chunk_path, format="wav")
        segment_paths.append(chunk_path)

    return segment_paths

def cleanup_files(paths: list[str])-> None:
    """ Xoa cac file tam sau khi xu li xong"""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
