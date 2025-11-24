from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

MODEL_SIZE = "small"       # "tiny", "base", "small", "medium", "large-v3"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2         # seconds

CHUNK_FRAMES = SAMPLE_RATE * CHUNK_DURATION

def main():
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print("Loading model...")
    print("Listening...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            audio, _ = stream.read(CHUNK_FRAMES)
            mono = audio.flatten().astype(np.float32)

            segments, _ = model.transcribe(mono, language="en", beam_size=1)

            for seg in segments:
                text = seg.text.strip()
                if text:
                    print("> ", text)


if __name__ == "__main__":
    main()
