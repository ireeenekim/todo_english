from src.voice_to_text import voice_to_text
from src.text_processing import process_text
from src.sns_generation import generate_sns_post

def main(audio_file_path):
    # Step 1: Convert voice to text
    conversation_text = voice_to_text(audio_file_path)
    print("Conversation Text:", conversation_text)
    
    # Step 2: Process the text
    processed_text = process_text(conversation_text)
    print("Processed Text:", processed_text)
    
    # Step 3: Generate SNS post
    sns_post = generate_sns_post(processed_text)
    print("SNS Post:", sns_post)

if __name__ == "__main__":
    audio_file_path = "./hello.wav"
    main(audio_file_path)
