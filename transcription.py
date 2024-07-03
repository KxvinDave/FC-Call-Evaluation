import whisper
import torch
class WhisperTranscibe:
    def __init__(self, modelSize='large'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = whisper.load_model(modelSize, device=device)
    def transcribe(self, audioPath):
        results = self.model.transcribe(audioPath, task='translate')
        segments = results['segments']
        return segments