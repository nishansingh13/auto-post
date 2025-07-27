import time
import requests

session = requests.Session()
#stewei M:24nz88eyawr3
# Login to get session cookies
login_response = session.post("https://api.fakeyou.com/login", json={
    "username_or_email": "nishan_singh137",
    "password": "Nishan$1378"
})
print("🔐 Login status:", login_response.status_code)

# Submit TTS job
submit_response = session.post("https://api.fakeyou.com/tts/inference", json={
    "tts_model_token": "weight_ymc3szhr46qbfgbyed9fp33cf",  # Peter Griffin voice
    "uuid_idempotency_token": str(time.time()),
"inference_text": (
    "Heeey Stewie...\n"
    "What's goin' on, buddy?\n"
    "It's me... Peter Griffin... haha.\n"
    "Lois... where's Quagmire?\n"
    "I wanna go down to the Drunken Clam...\n"
    "And drink a Pawtucket Patriot...\n"
    "With the boys.\n"
    "Heh...\n"
    "Heh...\n"
    "Heh."
)





})

job_data = submit_response.json()
print("🚀 Submit TTS job response:", job_data)

if not job_data.get("success"):
    print("❌ Failed to submit TTS job.")
    exit()

job_token = job_data["inference_job_token"]

# Poll for result
print("⏳ Polling for TTS generation...")
for i in range(30):
    poll_response = session.get(f"https://api.fakeyou.com/tts/job/{job_token}")
    poll_data = poll_response.json()
    state = poll_data["state"]["status"]

    print(f"⌛ Poll {i+1}: Status = {state}")
    if state == "complete_success":
        path = poll_data["state"].get("maybe_public_bucket_wav_audio_path")
        if path:
            url = f"https://cdn-2.fakeyou.com{path}"
            print("⬇️ Downloading from:", url)
            audio = session.get(url)
            with open("peter_voice.wav", "wb") as f:
                f.write(audio.content)
            print("✅ Saved: peter_voice.wav")
        else:
            print("❌ No audio path returned.")
        break
    elif state == "complete_failure":
        print("❌ TTS generation failed.")
        break

    time.sleep(2)
else:
    print("❌ Timed out.")
