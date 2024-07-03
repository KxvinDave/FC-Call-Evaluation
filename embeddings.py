import librosa
from pyannote.audio import Audio
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transcription import WhisperTranscibe

class Embeddings:
    def __init__(self, type='facebook/wav2vec2-large-960h-lv60', device='cpu'):
        self.processor = Wav2Vec2Processor.from_pretrained(type)
        self.model = Wav2Vec2Model.from_pretrained(type)
        self.transcibe = WhisperTranscibe()
        self.device = device
    def SegmentEmbeddings(self, audioPath, segment):
        Start = segment['start']
        End = segment['end']
        audio, sr = librosa.load(audioPath, sr=16000, offset=Start, duration=End-Start)
        waveform = torch.tensor(audio)
        inpVal = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
        inpVal = inpVal.to(self.device)
        with torch.no_grad():
            embeddings = self.model(inpVal).last_hidden_state
        aggregated = embeddings.cpu().mean(dim=1).squeeze().numpy()
        return aggregated
    def getEmbeddings(self, audioPath):
        segments = self.transcibe.transcribe(audioPath)
        embeddings = []
        for segment in segments:
            embeddings.append(self.SegmentEmbeddings(audioPath, segment))
        return embeddings