from moviepy import *
import os
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import whisper
from concurrent.futures import ThreadPoolExecutor

# Configuration
output_dir = "output"

def natural_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

audio_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".wav")], key=natural_sort_key)
minecraft_video = "minecraft_video.mp4"
final_output = "final.mp4"

peter_image = "peter_griffin.png"
stewie_image = "stewie.png"

# Load Whisper model once (auto-detect CUDA, fallback to CPU)
import torch
if torch.cuda.is_available():
    whisper_model = whisper.load_model("medium", device="cuda")
    print("[INFO] Using GPU (CUDA) for Whisper model.")
else:
    whisper_model = whisper.load_model("large-v3", device="cpu")
    print("[INFO] CUDA not available, using CPU for Whisper model.")

def get_whisper_words(audio_path):
    result = whisper_model.transcribe(audio_path, word_timestamps=True, language='en')
    words = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
    return words

background = VideoFileClip(minecraft_video)
total_audio_duration = sum(AudioFileClip(os.path.join(output_dir, f)).duration for f in audio_files)
if background.duration < total_audio_duration:
    loops_needed = int(total_audio_duration / background.duration) + 1
    background = concatenate_videoclips([background] * loops_needed)
target_width = 1080
target_height = 1920
video_aspect = background.w / background.h
target_aspect = target_width / target_height
if video_aspect > target_aspect:
    new_width = int(background.h * target_aspect)
    x_center = background.w // 2
    background = background.cropped(x_center=x_center, width=new_width)
else:
    new_height = int(background.w / target_aspect)
    y_center = background.h // 2
    background = background.cropped(y_center=y_center, height=new_height)
background = background.resized((target_width, target_height))

duration_accum = 0
clips = []

# Preload font and images
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 55)
except:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 55)
    except:
        font = ImageFont.load_default()
peter_img = ImageClip(peter_image).resized(height=700) if os.path.exists(peter_image) else None
stewie_img = ImageClip(stewie_image).resized(height=700) if os.path.exists(stewie_image) else None

def create_custom_text_clip(words, current_word_idx, duration, font_size=55):
    img_width, img_height = 1080, 800
    img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    left_margin = 150
    right_margin = 150
    usable_width = img_width - left_margin - right_margin
    # Use preloaded font
    used_font = font
    max_words_per_line = 3
    lines = []
    current_line = []
    for i, word in enumerate(words):
        current_line.append((word, i))
        if len(current_line) == max_words_per_line or i == len(words) - 1:
            lines.append(current_line)
            current_line = []
    line_height = 80
    start_y = 80
    for line_idx, line in enumerate(lines):
        line_text = " ".join([word for word, _ in line])
        bbox = draw.textbbox((0, 0), line_text, font=used_font)
        text_width = bbox[2] - bbox[0]
        x_start = left_margin + (usable_width - text_width) // 2
        y_pos = start_y + line_idx * line_height
        current_x = x_start
        for word, word_idx in line:
            color = (255, 0, 0, 255) if word_idx == current_word_idx else (0, 0, 0, 255)
            stroke_width = 3
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((current_x + dx, y_pos + dy), word, font=used_font, fill=(255, 255, 255, 255))
            draw.text((current_x, y_pos), word, font=used_font, fill=color)
            word_bbox = draw.textbbox((0, 0), word + " ", font=used_font)
            current_x += word_bbox[2] - word_bbox[0]
    img_array = np.array(img)
    return ImageClip(img_array, duration=duration)

def make_whisper_caption(words_with_times, total_duration, chunk_size=4):
    all_clips = []
    n = len(words_with_times)
    idx = 0
    while idx < n:
        chunk_words = words_with_times[idx:idx+chunk_size]
        chunk_texts = [w["text"] for w in chunk_words]
        chunk_start = chunk_words[0]["start"]
        chunk_end = min(chunk_words[-1]["end"], total_duration)
        chunk_duration = max(0, chunk_end - chunk_start)
        if chunk_duration <= 0:
            idx += chunk_size
            continue
        # Plain text, no highlight
        text_clip = create_custom_text_clip(chunk_texts, -1, chunk_duration, font_size=50)  # -1 disables highlight
        text_clip = text_clip.with_position(('center', 500)).with_start(chunk_start)
        all_clips.append(text_clip)
        idx += chunk_size
    return CompositeVideoClip(all_clips, size=(1080, 1920)).with_duration(total_duration)

def create_character_overlay(speaker, duration):
    if speaker.lower() == "peter":
        img_clip = peter_img
        position = (80, 'bottom')
    else:
        img_clip = stewie_img
        position = (500, 'bottom')
    if img_clip:
        return img_clip.with_duration(duration).with_position(position).with_opacity(1.0)
    else:
        return None

def process_audio_file(i, filename, duration_accum):
    audio_path = os.path.join(output_dir, filename)
    audio_clip = AudioFileClip(audio_path)
    words_with_times = get_whisper_words(audio_path)
    if not words_with_times:
        print(f"No words found by Whisper in {filename}, skipping caption.")
        return None, duration_accum
    speaker = 'peter' if i % 2 == 0 else 'stewie'
    text_clip = make_whisper_caption(words_with_times, audio_clip.duration, chunk_size=4)  # Use chunked plain text
    character_overlay = create_character_overlay(speaker, audio_clip.duration)
    video_segment = background.subclipped(duration_accum, duration_accum + audio_clip.duration).without_audio()
    video_segment = video_segment.with_audio(audio_clip)
    layers = [video_segment, text_clip]
    if character_overlay:
        layers.append(character_overlay)
    composite = CompositeVideoClip(layers)
    return composite, duration_accum + audio_clip.duration

# Parallel render audio files (threaded, but limit to 2 workers for safety)
results = []
from time import time
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    duration_accum = 0
    for i, filename in enumerate(audio_files):
        futures.append(executor.submit(process_audio_file, i, filename, duration_accum))
        # Update duration_accum for next file (serially, since video segments must be sequential)
        audio_path = os.path.join(output_dir, filename)
        audio_clip = AudioFileClip(audio_path)
        duration_accum += audio_clip.duration
    for future in futures:
        result, _ = future.result()
        if result:
            results.append(result)

final_video = concatenate_videoclips(results)
final_video.write_videofile(final_output, codec='libx264', audio_codec='aac', threads=8)
