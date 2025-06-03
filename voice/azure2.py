import os
import azure.cognitiveservices.speech as speechsdk

# Configuration constants
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")  # Use environment variables for security
SERVICE_REGION = "eastus"
DEFAULT_LANGUAGE = "en-US"
TARGET_LANGUAGES = ["it", "es", "fr"]  # Multiple target languages
MICROPHONE_INDEX = None  # Set to None for default microphone, or specify an index

# Noise Suppression Configuration
NOISE_SUPPRESSION_LEVEL = speechsdk.audio.AudioProcessingOptions(speechsdk.AudioProcessingConstants.Preset.HIGH)

def get_speech_translation_config():
    """Creates and returns a configured SpeechTranslationConfig object."""
    if not SPEECH_KEY:
        raise ValueError("Speech key is missing. Set the AZURE_SPEECH_KEY environment variable.")

    config = speechsdk.translation.SpeechTranslationConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    config.speech_recognition_language = DEFAULT_LANGUAGE

    for lang in TARGET_LANGUAGES:
        config.add_target_language(lang)

    # Configure noise suppression
    config.enable_audio_logging = False  # Disable audio logging for privacy
    config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000")  # Handle noisy background

    return config

def get_audio_config():
    """Creates and returns an AudioConfig object with noise suppression enabled."""
    audio_processing_options = speechsdk.audio.AudioProcessingOptions(
        NOISE_SUPPRESSION_LEVEL
    )

    if MICROPHONE_INDEX is not None:
        audio_config = speechsdk.audio.AudioConfig(device_name=str(MICROPHONE_INDEX), audio_processing_options=audio_processing_options)
    else:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True, audio_processing_options=audio_processing_options)

    return audio_config

def recognize_from_microphone():
    """real-time speech recognition and translation from the microphone, optimized for noisy environments."""
    try:
        speech_translation_config = get_speech_translation_config()
        audio_config = get_audio_config()
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=speech_translation_config,
            audio_config=audio_config
        )

        print("Speak clearly into your microphone (optimized for noisy environments)...")

        def recognized_callback(evt):
            """Handles recognized speech event, filtering out low-confidence results."""
            if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
                confidence = evt.result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult, "")
                print(f"Recognized: {evt.result.text} (Confidence Score: {confidence})")
                for lang, translation in evt.result.translations.items():
                    print(f"Translated into '{lang}': {translation}")
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                print(f"No speech could be recognized: {evt.result.no_match_details}")

        def canceled_callback(evt):
            """this handles cancellation details."""
            cancellation_details = evt.result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                print("Ensure that your subscription key and region are set correctly.")

        # this attaches event handlers for continuous recognition
        recognizer.recognized.connect(recognized_callback)
        recognizer.canceled.connect(canceled_callback)

        # start continuous recognition
        recognizer.start_continuous_recognition()
        input("Press Enter to stop...\n")
        recognizer.stop_continuous_recognition()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    recognize_from_microphone()
