import os
import time
import uuid
import threading
import logging
from scipy.io import wavfile
try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Azure SDK not installed. Please install with 'pip install azure-cognitiveservices-speech'")
    import sys
    sys.exit(1)
# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Secure API key retrieval
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")  # Store key securely in environment variable
SERVICE_REGION = "eastus"
WAV_FILE = "sample.wav"  # Ensure file exists in the directory
def initialize_conversation_transcriber():
    """THis will initialize and configure the Azure conversation transcriber."""
    if not SPEECH_KEY:
        logging.error("Azure Speech Key is missing. Set the AZURE_SPEECH_KEY environment variable.")
        raise ValueError("Azure Speech Key is required.")

    # Speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    speech_config.set_property_by_name("ConversationTranscriptionInRoomAndOnline", "true")
    speech_config.set_property_by_name("DifferentiateGuestSpeakers", "true")
    # Audio settings
    channels = 1  # Set to 1 for mono-channel audio (better for voice recognition)
    bits_per_sample = 16
    samples_per_second = 16000
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second, bits_per_sample, channels)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    # Unique conversation ID
    conversation_id = str(uuid.uuid4())
    conversation = speechsdk.transcription.Conversation(speech_config, conversation_id)
    transcriber = speechsdk.transcription.ConversationTranscriber(audio_config)
    return transcriber, conversation, stream

def handle_transcription_event(evt):
    """Handles and logs transcription events."""
    if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
        speaker = evt.result.speaker_id if evt.result.speaker_id else "Unknown"
        logging.info(f"Speaker {speaker}: {evt.result.text}")

def handle_cancellation_event(evt):
    """Handles and logs cancellation events."""
    logging.error(f"Transcription Canceled: {evt.result.cancellation_details.reason}")
    if evt.result.cancellation_details.reason == speechsdk.CancellationReason.Error:
        logging.error(f"Error details: {evt.result.cancellation_details.error_details}")

def start_transcription():
    """Handles streaming of audio file and transcription process."""
    try:
        transcriber, conversation, stream = initialize_conversation_transcriber()
        transcriber.transcribed.connect(handle_transcription_event)
        transcriber.session_stopped.connect(lambda evt: logging.info("Session Stopped"))
        transcriber.canceled.connect(handle_cancellation_event)

        # Add participants dynamically
        participants = [
            speechsdk.transcription.Participant("katie@example.com", "en-us"),
            speechsdk.transcription.Participant("stevie@example.com", "en-us")
        ]
        for participant in participants:
            conversation.add_participant_async(participant)

        transcriber.join_conversation_async(conversation).get()
        transcriber.start_transcribing_async()

        # Read and stream the WAV file in chunks
        def stream_audio():
            try:
                _, wav_data = wavfile.read(WAV_FILE)
                for i in range(0, len(wav_data), 4096):  # Stream in chunks
                    stream.write(wav_data[i:i+4096].tobytes())
                    time.sleep(0.1)  # Simulate real-time processing
                stream.close()
            except FileNotFoundError:
                logging.error(f"Audio file {WAV_FILE} not found.")
                transcriber.stop_transcribing_async()
            except Exception as e:
                logging.error(f"Error streaming audio: {e}")
                transcriber.stop_transcribing_async()

        # Run audio streaming in a separate thread
        audio_thread = threading.Thread(target=stream_audio)
        audio_thread.start()

        # Wait for transcription to complete
        audio_thread.join()
        transcriber.stop_transcribing_async()

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    start_transcription()
