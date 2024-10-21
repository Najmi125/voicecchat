import whisper
import os
from groq import Groq
from gtts import gTTS
import gradio as gr

# Load the pre-trained Whisper model
model = whisper.load_model("base")

# api
GROQ_API_KEY = "APNI KHUD KI API IS MA DAL DE"
#client = Groq(api_key=GROQ_API_KEY)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

# Function to get response from Groq's LLM
def get_llm_response(user_text):
    #client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_text,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to convert text to speech using gTTS
def convert_text_to_speech(text, output_path="response.mp3"):
    tts = gTTS(text)
    tts.save(output_path)
    return output_path

# Main pipeline function for the Gradio interface
def chatbot_pipeline(audio_input):
    # Step 1: Transcribe audio to text
    transcribed_text = transcribe_audio(audio_input)
    
    # Step 2: Get LLM response for the transcribed text
    llm_response = get_llm_response(transcribed_text)
    
    # Step 3: Convert LLM response to speech
    audio_output = convert_text_to_speech(llm_response)
    
    return audio_output

# Gradio interface setup
gr.Interface(
    fn=chatbot_pipeline, 
    inputs=gr.Audio(type="filepath"),  # Use gr.Audio without the 'source' argument
    outputs=gr.Audio(type="filepath")  # The output will also be an audio file path
).launch()
