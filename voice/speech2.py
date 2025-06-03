"""
transcription
"""

import time
import uuid

from scipy.io import wavfile

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Fail
    """)
    import sys
    sys.exit(1)

speech_key, service_region = "", "eastus"

conversationfilename = "sample"

def conversation_transcription_differentiate_speakers():
    """differentiates speakers using conversation transcription service"""
    # Creates speech configuration with subscription information
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.set_property_by_name("ConversationTranscriptionInRoomAndOnline", "true")
    speech_config.set_property_by_name("DifferentiateGuestSpeakers", "true")

    channels = 8
    bits_per_sample = 16
    samples_per_second = 16000

    # Create audio configuration using the push stream
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second, bits_per_sample, channels)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # Conversation identifier is required when creating conversation.
    conversation_id = str(uuid.uuid4())
    conversation = speechsdk.transcription.Conversation(speech_config, conversation_id)
    transcriber = speechsdk.transcription.ConversationTranscriber(audio_config)

    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous transcription upon receiving an event `evt`"""
        print('CLOSING {}'.format(evt))
        nonlocal done
        done = True

    # Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(lambda evt: print('TRANSCRIBED: {}'.format(evt)))
    transcriber.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    transcriber.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    transcriber.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)

    katie = speechsdk.transcription.Participant("katie@example.com", "en-us")
    stevie = speechsdk.transcription.Participant("stevie@example.com", "en-us")

    conversation.add_participant_async(katie)
    conversation.add_participant_async(stevie)
    transcriber.join_conversation_async(conversation).get()
    transcriber.start_transcribing_async()

    # Read the whole wave files at once and stream it to sdk
    _, wav_data = wavfile.read(conversationfilename)
    stream.write(wav_data.tobytes())
    stream.close()
    while not done:
        time.sleep(.5)

    transcriber.stop_transcribing_async()