'''
transcriber_whisper.py
----------------------
Dùng model whisper (OpenAI, chạy local) để chuyển audio thành văn bản.
Model chỉ được load 1 lần và tái sử dụng cho mọi request.

Lưu ý:
- Cần set PYTORCH_ENABLE_MPS_FALLBACK = 1 TRƯỚC khi import torch
 (Một số toán tử Whisper chưa hỗ trợ đầy đủ trên MPS)
'''

import os
import ssl

#Tat verify (can thiet tren mot so may khi download model Whisper
ssl._create_default_https_context = ssl._create_unverified_context

#Set bien moi truong cho MPS fallback (PHAI DAT truoc khi import torch)
os.environ.setdefault(key="PYTORCH_ENABLE_MPS_FALLBACK", value = 1)

import torch
import whisper

# ===============================
# CẤU HÌNH
# ===============================
# Các model Whisper: "tiny", "base", "small", "medium", "large-v3"
# Model lớn hơn -> chính xác hơn nhưng chaamj hơn, ton RAM hon
DEFAULT_MODEL_SIZE = "medium"

# =================================
# HAM CHON DEVICE
# =================================
def get_device()->str:
    '''
    Tu dong chon thiet bi tot nhat: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    :return:
    '''
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEFAULT_DEVICE = get_device()

# ===============================
# LOAD MODEL (LAZY LOAD - chi load 1 lan)
# ===============================

model = None

def get_model(model_size:str = DEFAULT_MODEL_SIZE, device: str = None):
    '''
    Load model Whisper: dung bien global _model de chi load 1 lan duy nhat

    :param model_size: kich thuoc model ("tiny", "base", "small", "medium", "large-v3"
    :param device: thiet bi chay ("cpu, "cuda", "mps")
    :return:
        Model Whisper da load
    '''
    global _model
    if _model is None:
        device = device or DEFAULT_DEVICE
        print(f'[transcribe] Dang load Whisper model{model_size} tren device {device}...' )
        _model = whisper.load_model(model_size, device = device)
        print('[transcribe] Model da san sang')
    return _model

# ===============================
# TRANSCRIBE 1 DOAN AUDIO
# ===============================
def transcribe_segment(segment_path: str, language:str=None)-> str:
    '''
    Chuyen 1 doan audio thanh van ban
    :param segment_path: duong dan file audio doan nho (wav).
    :param language: ma ngon ngu ("vi", "en"...). None -> Whisper tu nhan dien.
    :return: Van ban da transcribe
    '''
    model = get_model()
    use_fp16 = model.device.type == "cuda"
    result = model.transcribe(segment_path, language = language, fp16=use_fp16)
    return result.get("text", "").strip()

# ===============================
# TRANSCRIBE NHIEU DOAN VA GHEP LAI
# ===============================

def transcrie_segments(segment_paths: list[str], language:str=None, progress_callback=None)-> str:
    '''
    Transcribe nhieu doan audio lien tiep va ghep lai thanh 1 van ban day du.
    :param segment_paths: danh sach duong dan cac doan audio (dung thu tu).
    :param language: ma ngon ngu. None de tu nhan dien.
    :param preogress_callback: ham callback(current_index, total) de bao tien do.
    :return: Van ban transcript day du, cac doan noi voi nhau bang dau xuong dong.
    '''
    full_text_parts = []
    total = len(segment_paths)

    for idx, path in enumerate(segment_paths, start = 1):
        text = transcribe_segment(path, language = language)
        full_text_parts.append(text)
        if progress_callback:
            progress_callback(idx, total)

    return "\n".join(full_text_parts)
