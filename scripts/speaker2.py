import time
import requests
import os

# Configuration
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Replace with your actual credentials and model tokens
USERNAME = "nishan_singh137"
PASSWORD = "Nishan$1378"
TTS_MODELS = {
    "peter": "weight_ymc3szhr46qbfgbyed9fp33cf",
    "stewie": "TM:24nz88eyawr3"  # Replace with correct one if this is outdated
}

# Dialogue script (React vs Angular debate)
dialogue = [
    ("peter", "Hey Stewie, I've been using this Copilot thing. It writes all my code for me!"),
    ("stewie", "All your code? Good heavens, Peter. You mean you're just a typist now?"),
    ("peter", "No no, I just hit tab a few times, and boom! App done."),
    ("stewie", "Let me guess, you deployed without even understanding the logic?"),
    ("peter", "Heh, well... it ran, didn't it?"),
    ("stewie", "Peter, Copilot isn't supposed to *replace* thinking. It's like using a calculator without knowing math."),
    ("peter", "So you're saying I should maybe learn what I'm building...?"),
    ("stewie", "Yes, Peter. Or one day you'll ask Copilot to build Skynet."),
]


# Start session
session = requests.Session()

# Login
login_response = session.post("https://api.fakeyou.com/login", json={
    "username_or_email": USERNAME,
    "password": PASSWORD
})
print("🔐 Login status:", login_response.status_code)
if login_response.status_code != 200:
    print("❌ Failed to login.")
    exit()

# Function to synthesize and save voice
def synthesize_and_save(speaker, text, index):
    print(f"\n🗣️ Generating voice for {speaker} (line {index+1})...")

    submit_response = session.post("https://api.fakeyou.com/tts/inference", json={
        "tts_model_token": TTS_MODELS[speaker],
        "uuid_idempotency_token": str(time.time()),
        "inference_text": text
    })
    
    job_data = submit_response.json()
    if not job_data.get("success"):
        print("❌ Failed to submit TTS job.")
        return

    job_token = job_data["inference_job_token"]

    # Poll for result
    for i in range(40):
        poll_response = session.get(f"https://api.fakeyou.com/tts/job/{job_token}")
        poll_data = poll_response.json()
        state = poll_data["state"]["status"]
        print(f"⌛ Poll {i+1}: Status = {state}")
        if state == "complete_success":
            path = poll_data["state"].get("maybe_public_bucket_wav_audio_path")
            if path:
                url = f"https://cdn-2.fakeyou.com{path}"
                audio = session.get(url)
                file_path = os.path.join(output_dir, f"{index+1}{speaker}.wav")
                with open(file_path, "wb") as f:
                    f.write(audio.content)
                print(f"✅ Saved: {file_path}")
            else:
                print("❌ No audio path returned.")
            break
        elif state == "complete_failure":
            print("❌ TTS generation failed.")
            break
        time.sleep(2)
    else:
        print("❌ Timed out.")

# Iterate through the dialogue
for i, (speaker, line) in enumerate(dialogue):
    synthesize_and_save(speaker, line, i)
