from moviepy import *
import os
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import whisper

# Configuration
output_dir = "output"

def natural_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# Load all audio files from the output directory and sort naturally
audio_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".wav")], key=natural_sort_key)
minecraft_video = "minecraft_video.mp4"  
final_output = "final.mp4"

peter_image = "peter_griffin.png"  # Path to Peter image
stewie_image = "stewie.png"  # Path to Stewie image

# Load Whisper model once
whisper_model = whisper.load_model("large", device="cpu")
 # You can use "small" or "medium" for better accuracy
 # here we are adding the whisper model to transcribe audio files

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

# Load Minecraft gameplay video and crop to 9:16 aspect ratio
background = VideoFileClip(minecraft_video)

# Calculate total audio duration first
total_audio_duration = sum(AudioFileClip(os.path.join(output_dir, f)).duration for f in audio_files)

# If video is shorter than total audio, loop it
if background.duration < total_audio_duration:
    loops_needed = int(total_audio_duration / background.duration) + 1
    background = concatenate_videoclips([background] * loops_needed)

# Crop to 9:16 aspect ratio (1080x1920 for vertical video)
target_width = 1080
target_height = 1920

# Calculate crop dimensions to maintain aspect ratio
video_aspect = background.w / background.h
target_aspect = target_width / target_height

if video_aspect > target_aspect:
    # Video is wider, crop width
    new_width = int(background.h * target_aspect)
    x_center = background.w // 2
    background = background.cropped(x_center=x_center, width=new_width)
else:
    # Video is taller, crop height  
    new_height = int(background.w / target_aspect)
    y_center = background.h // 2
    background = background.cropped(y_center=y_center, height=new_height)

# Resize to final dimensions
background = background.resized((target_width, target_height))

duration_accum = 0
clips = []

# PIL-based text rendering function
def create_custom_text_clip(words, highlight_idx, duration, font_size=65):
    # Create image with transparent background
    img_width, img_height = 1080, 800
    img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Add much larger margins to prevent text overflow
    left_margin = 150   # Increased from 80 to 150px
    right_margin = 150  # Increased from 80 to 150px  
    usable_width = img_width - left_margin - right_margin  # 780px usable width (down from 920px)
    
    # Try to load Proxima Nova Semibold font, fallback to DejaVuSans-Bold or Arial
    # try:
    font = ImageFont.truetype("./fonnts.com-ProximaNova-Semibold.ttf", font_size)

    
    # Break words into lines (max 3 words per line due to smaller width)
    max_words_per_line = 3  # Reduced from 4 to 3 due to smaller usable width
    lines = []
    current_line = []
    
    for i, word in enumerate(words):
        current_line.append((word, i))
        if len(current_line) == max_words_per_line or i == len(words) - 1:
            lines.append(current_line)
            current_line = []
    
    # Draw each line
    line_height = 80
    start_y = 80
    
    for line_idx, line in enumerate(lines):
        # Calculate line text for centering within the margin area
        line_text = " ".join([word for word, _ in line])
        bbox = draw.textbbox((0, 0), line_text, font=font)
        text_width = bbox[2] - bbox[0]
        
        # Center the text within the usable width (between larger margins)
        x_start = left_margin + (usable_width - text_width) // 2
        y_pos = start_y + line_idx * line_height
        
        # Draw each word in the line
        current_x = x_start
        for word, word_idx in line:
            # Light green for the highlighted word, white for previous
            if word_idx == highlight_idx:
                color = (102, 255, 102, 255)  # Light green
                # Extra bold: draw more times for highlight
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        draw.text((current_x + dx, y_pos + dy), word, font=font, fill=color)
            else:
                color = (255, 255, 255, 255)  # White
                # Draw black shadow for visibility
                draw.text((current_x + 2, y_pos + 2), word, font=font, fill=(0, 0, 0, 200))
                # Draw normal white text
                draw.text((current_x, y_pos), word, font=font, fill=color)
            
            # Move to next word position
            word_bbox = draw.textbbox((0, 0), word + " ", font=font)
            current_x += word_bbox[2] - word_bbox[0]
    
    # Convert PIL image to numpy array for MoviePy
    img_array = np.array(img)
    
    return ImageClip(img_array, duration=duration)

def make_whisper_caption(words_with_times, total_duration, chunk_size=3):
    all_clips = []
    words = [fix_transcript(w["text"]) for w in words_with_times]
    n = len(words)
    idx = 0
    while idx < n:
        chunk_end_idx = min(idx + chunk_size, n)
        chunk_words = words[idx:chunk_end_idx]
        chunk_times = words_with_times[idx:chunk_end_idx]
        for j, word in enumerate(chunk_times):
            display_words = chunk_words[:j+1]
            highlight_idx = j  # Highlight the current word in the chunk
            word_start = word["start"]
            word_end = min(word["end"], total_duration)
            word_duration = max(0, word_end - word_start)
            if word_duration <= 0:
                continue
            text_clip = create_custom_text_clip(display_words, highlight_idx, word_duration, font_size=70)
            text_clip = text_clip.with_position(('center', 500)).with_start(word_start)
            all_clips.append(text_clip)
        idx += chunk_size
    return CompositeVideoClip(all_clips, size=(1080, 1920)).with_duration(total_duration)

def create_character_overlay(speaker, duration):
    if speaker.lower() == "peter":
        img_path = peter_image
        position = (80, 'bottom')  # Moved Peter right from 30 to 80
    else:  # stewie
        img_path = stewie_image
        position = (500, 'bottom')  # Moved Stewie left from 'right' to 900px from left edge
    
    if os.path.exists(img_path):
        return (ImageClip(img_path)
                .resized(height=700)  # Much larger images (increased from 500 to 700)
                .with_duration(duration)
                .with_position(position)
                .with_opacity(1.0))  # Full opacity for better visibility
    else:
        print(f"Warning: {img_path} not found!")
        return None

corrections = {
    "So wait": "Stewie",
    "Epenmarn": "MERN",
    "Should": "Stewie",
    "we":"I",
    "Worth":"worse",
    "their":"they"
    # Add more as needed
}

def fix_transcript(text):
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text


# Process each audio file (no manual dialogue)
for i, filename in enumerate(audio_files):
    audio_path = os.path.join(output_dir, filename)
    audio_clip = AudioFileClip(audio_path)
    # Run Whisper on this audio file
    words_with_times = get_whisper_words(audio_path)
    if not words_with_times:
        print(f"No words found by Whisper in {filename}, skipping caption.")
        continue
    # Use first word as speaker guess (optional, or set to 'peter'/'stewie' by order)
    speaker = 'peter' if i % 2 == 0 else 'stewie'
    text_clip = make_whisper_caption(words_with_times, audio_clip.duration)  # No chunk_size
    character_overlay = create_character_overlay(speaker, audio_clip.duration)

    video_segment = background.subclipped(duration_accum, duration_accum + audio_clip.duration).without_audio()
    video_segment = video_segment.with_audio(audio_clip)

    # Composite video with text and character (character only appears when speaking)
    layers = [video_segment, text_clip]
    if character_overlay:
        layers.append(character_overlay)
    
    composite = CompositeVideoClip(layers)
    clips.append(composite)

    duration_accum += audio_clip.duration

# Concatenate and export
final_video = concatenate_videoclips(clips)
final_video.write_videofile(final_output, codec='libx264', audio_codec='aac')
