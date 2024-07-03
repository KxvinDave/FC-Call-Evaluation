import sys
from V2A import convert
from transcription import WhisperTranscibe
from diarize import diarize
from embeddings import Embeddings
from LLM import EvaluteNeeds
from identify import SpeakerIdentification

def main(videoPath, audioPath, context):
    #Step1: Convert to audio
    convert(videoPath, audioPath)
    print("Conversion completed!")
    #Step2: Transcribe
    transcriber = WhisperTranscibe()
    print("Transcribing... ")
    segments = transcriber.transcribe(audioPath)

    print("Transcription generated")

    #Generate embeddings
    embedder = Embeddings()
    embeddings = embedder.getEmbeddings(segments)

    #Diarize speakers
    diarised = diarize(embeddings, segments)

    #Classify speakers
    speakerID = SpeakerIdentification()
    transcript = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in diarised])
    speakerRoles = speakerID.identification(transcript)
    finalTranscript = speakerID.replaceTags(transcript, speakerRoles)

    #Evaluate needs using LLM
    evaluator = EvaluteNeeds()
    needs = evaluator.getNeeds(finalTranscript, context)

    return needs

if __name__ == "__main__":
    if len(sys.args) != 4:
        print("Usage: python main.py <video_path> <audio_path> <context>")
        sys.exit(1)
    videoPath = sys.argv[1]
    audioPath = sys.argv[2]
    context = sys.argv[3]

    main(videoPath, audioPath, context)