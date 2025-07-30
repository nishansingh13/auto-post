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
    ("peter", "Stewie! I just signed up for twelve new streaming services! This free trial button was so easy to click!"),
    ("stewie", "You gullible simpleton! You are a victim of a dark pattern! It is deliberate manipulation, designed to trick you!"),
    ("peter", "Trick me? Like when Lois says there is pie, but it is really kale? Oof."),
    ("stewie", "Worse! They hide the cancel button, auto-enroll you, or make opting out incredibly difficult! It is psychological warfare on your wallet!"),
    ("peter", "So, that tiny little checkbox that said share my DNA with aliens was... a trick?"),
    ("stewie", "Precisely! They exploit cognitive biases. You are trying to do one thing, and they steer you to another!"),
    ("peter", "But why would an app want to trick me? Apps are supposed to be my friends!"),
    ("stewie", "To extract more money, more data, more engagement! They are not your friends, Peter; they are digital con artists!"),
    ("peter", "Aw, nuts. So my free trial for the Extreme Squirrel Watching Channel is not actually free forever?"),
    ("stewie", "No, you nincompoop! It is a monthly charge! Now, if you will excuse me, I need to design an interface to stop your impulse purchases!"),
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
def synthesize_and_save(speaker, text, index, max_retries=3):
    for attempt in range(1, max_retries + 1):
        print(f"\n🗣️ Generating voice for {speaker} (line {index+1}), attempt {attempt}...")

        submit_response = session.post("https://api.fakeyou.com/tts/inference", json={
            "tts_model_token": TTS_MODELS[speaker],
            "uuid_idempotency_token": str(time.time()),
            "inference_text": text
        })
      
        job_data = submit_response.json()
        if not job_data.get("success"):
            print("❌ Failed to submit TTS job.")
            continue

        job_token = job_data["inference_job_token"]

        # Poll for result
        for i in range(40):
            # time.sleep(5)
            poll_response = session.get(f"https://api.fakeyou.com/tts/job/{job_token}")
            poll_data = poll_response.json()
            state = poll_data["state"]["status"]
            print(f"⌛ Poll {i+1}: Status = {state}")
            if state == "complete_success":
                path = poll_data["state"].get("maybe_public_bucket_wav_audio_path")
                if path:
                    url = f"https://cdn-2.fakeyou.com{path}"
                    audio = session.get(url)
                    file_path = os.path.join(output_dir, f"{index+1}.wav")
                    with open(file_path, "wb") as f:
                        f.write(audio.content)
                    print(f"✅ Saved: {file_path}")
                else:
                    print("❌ No audio path returned.")
                return  # Success, exit function
            elif state in ("complete_failure", "cancelled"):
                print(f"❌ TTS generation failed or cancelled (state: {state}). Retrying...")
                break  # Break polling loop and retry
            time.sleep(2)
        else:
            print("❌ Timed out. Retrying...")
    print(f"❌ All {max_retries} attempts failed for line {index+1}.")

# Iterate through the dialogue
for i, (speaker, line) in enumerate(dialogue):
    synthesize_and_save(speaker, line, i)
