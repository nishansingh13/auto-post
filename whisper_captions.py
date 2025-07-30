#!/usr/bin/env python3
"""
Integrated Whisper Caption Generator
Only shows the word currently being spoken, one at a time, in sync with the audio.
"""

import whisper
import os
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip

def create_captions_with_whisper():
    # Your existing dialogue from speaker2.py
    dialogue = [
        ("peter", "Stewie! Did you see this? Aliens built the pyramids! It was on my feed!"),
        ("stewie", "Oh, for crying out loud. Your feed isn't just showing you what you want, you simpleton. It's molding your reality!"),
        ("peter", "Molding? Like Lois molds my brain when she talks about chores? Oof."),
        ("stewie", "The algorithm prioritizes content that keeps you engaged. Outrage, conspiracy, anything that triggers an emotional response!"),
        ("peter", "So it knows I like videos of people falling down stairs? And then it gives me more?"),
        ("stewie", "Exactly! It creates an echo chamber, reinforcing your existing biases until you think the Earth is flat!"),
        ("peter", "Wait, the Earth isn't flat? My feed told me it was a giant pizza!"),
        ("stewie", "This is precisely the problem! You're not scrolling, you're being scrolled! Your thoughts are no longer your own!"),
        ("peter", "So you mean my sudden urge to build a bunker is the internet's fault? Aw, nuts."),
        ("stewie", "It is, you gullible ape! And frankly, the bunker might be safer from your own online idiocy."),
    ]

    print("🤖 Loading Whisper...")
    model = whisper.load_model("base")
    video_path = "final.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    print("🎬 Loading video...")
    video = VideoFileClip(video_path)
    caption_clips = []
    for i, (speaker, text) in enumerate(dialogue):
        audio_file = f"output/{i+1}.wav"
        if not os.path.exists(audio_file):
            print(f"⚠️ Audio file not found: {audio_file}")
            continue
        print(f"🎵 Processing {speaker}: {audio_file}")
        result = model.transcribe(audio_file, word_timestamps=True)
        for segment in result['segments']:
            for word_info in segment.get('words', []):
                word = word_info['word'].strip()
                start = word_info['start']
                end = word_info['end']
                # Only show the current word, centered, one at a time
                text_clip = TextClip(
                    text=word,
                    font_size=60,
                    color='white',
                    size=(600, 120)
                ).with_position(('center', 'center')).with_start(start).with_duration(end - start)
                caption_clips.append(text_clip)
    print("🎞️ Adding captions to video...")
    final_video = CompositeVideoClip([video] + caption_clips)
    output_path = "final_with_captions.mp4"
    print(f"💾 Saving to {output_path}...")
    final_video.write_videofile(output_path, codec='libx264')
    print("✅ Done!")

if __name__ == "__main__":
    create_captions_with_whisper()