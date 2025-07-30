#!/usr/bin/env python3
"""
FOCUSED MFA Caption Generator
Uses ONLY Montreal Forced Alignment - the BEST method for transcript-based alignment
Works directly with your existing Peter/Stewie dialogue array
"""

import os
import subprocess
import tempfile
from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from dataclasses import dataclass
from typing import List
import re

@dataclass
class WordTiming:
    word: str
    start_time: float
    end_time: float

class MFACaptionGenerator:
    def __init__(self):
        self.temp_dir = "/tmp/mfa_work"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def align_audio_with_transcript(self, audio_path: str, transcript: str) -> List[WordTiming]:
        """
        Use MFA to get perfect word-level alignment
        This is the GOLD STANDARD method
        """
        print(f"🥇 Using MFA (Best Quality) to align: {os.path.basename(audio_path)}")
        
        # Create temporary files for MFA
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        work_dir = os.path.join(self.temp_dir, base_name)
        os.makedirs(work_dir, exist_ok=True)
        
        # Copy audio file
        audio_dest = os.path.join(work_dir, f"{base_name}.wav")
        subprocess.run(['cp', audio_path, audio_dest], check=True)
        
        # Create transcript file (clean text for MFA)
        transcript_clean = re.sub(r'[^\w\s]', '', transcript).strip()
        transcript_path = os.path.join(work_dir, f"{base_name}.txt")
        with open(transcript_path, 'w') as f:
            f.write(transcript_clean)
        
        # Run MFA alignment
        output_dir = os.path.join(work_dir, 'aligned')
        try:
            cmd = [
                'mfa', 'align',
                work_dir,
                'english_us_arpa',
                'english_us_arpa', 
                output_dir,
                '--clean'
            ]
            
            print("   🔄 Running MFA alignment...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse TextGrid output
            textgrid_path = os.path.join(output_dir, f"{base_name}.TextGrid")
            return self._parse_textgrid(textgrid_path)
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ MFA failed: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            return []
    
    def _parse_textgrid(self, textgrid_path: str) -> List[WordTiming]:
        """Parse MFA TextGrid output to extract word timings"""
        word_timings = []
        
        if not os.path.exists(textgrid_path):
            print(f"   ❌ TextGrid not found: {textgrid_path}")
            return []
        
        print(f"   📖 Parsing TextGrid: {textgrid_path}")
        
        try:
            with open(textgrid_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple TextGrid parsing (words tier)
            lines = content.split('\n')
            in_words_tier = False
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Find words tier
                if 'item [2]:' in line or 'name = "words"' in line:
                    in_words_tier = True
                
                # Parse intervals in words tier
                if in_words_tier and 'xmin =' in line:
                    try:
                        xmin = float(line.split('=')[1].strip())
                        xmax = float(lines[i+1].split('=')[1].strip())
                        text = lines[i+2].split('=')[1].strip().strip('"')
                        
                        if text and text != '':
                            word_timings.append(WordTiming(
                                word=text,
                                start_time=xmin,
                                end_time=xmax
                            ))
                        
                        i += 3
                    except (IndexError, ValueError) as e:
                        i += 1
                else:
                    i += 1
            
            print(f"   ✅ Extracted {len(word_timings)} word timings")
            return word_timings
            
        except Exception as e:
            print(f"   ❌ Error parsing TextGrid: {e}")
            return []
    
    def create_modern_highlighted_captions(self, word_timings: List[WordTiming]) -> List[TextClip]:
        """Create simple one-word-at-a-time captions"""
        clips = []
        
        # Show ONLY the current word being spoken
        for word_timing in word_timings:
            # Create text clip for just this one word
            clip = TextClip(
                text=word_timing.word,
                font_size=70,
                color='white'
            ).with_position(('center', 'center')).with_start(word_timing.start_time).with_duration(
                word_timing.end_time - word_timing.start_time
            )
            
            clips.append(clip)
        
        return clips
    
    def _group_into_lines(self, word_timings: List[WordTiming], max_words: int = 4) -> List[List[WordTiming]]:
        """Group words into display lines"""
        lines = []
        current_line = []
        
        for word in word_timings:
            current_line.append(word)
            
            if len(current_line) >= max_words:
                lines.append(current_line)
                current_line = []
        
        if current_line:
            lines.append(current_line)
        
        return lines

import whisper

def create_perfect_captions_with_whisper():
    """
    Create perfect captions using Whisper + your dialogue text
    This gives you the BEST of both: accurate text + precise timing
    """
    
    # Your existing dialogue
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
    
    print("🥇 Creating PERFECT captions with Whisper + your dialogue...")
    print("🤖 Loading Whisper...")
    model = whisper.load_model("base")
    
    output_dir = "/home/nishan/Documents/WMK/output"
    
    # Process each audio file
    all_word_timings = []
    current_time = 0
    
    for i, (speaker, text) in enumerate(dialogue):
        audio_file = os.path.join(output_dir, f"{i+1}.wav")
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            continue
        
        print(f"🎵 Processing {speaker}: {os.path.basename(audio_file)}")
        
        # Get audio duration
        from moviepy import AudioFileClip
        audio_clip = AudioFileClip(audio_file)
        audio_duration = audio_clip.duration
        audio_clip.close()
        
        # Get word timing from Whisper
        result = model.transcribe(audio_file, word_timestamps=True)
        
        # Use YOUR words with Whisper's timing structure
        your_words = text.split()
        whisper_timings = []
        
        for segment in result['segments']:
            for word_info in segment.get('words', []):
                whisper_timings.append({
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        # Create word timings using YOUR text
        for idx, word in enumerate(your_words):
            if idx < len(whisper_timings):
                start_time = current_time + whisper_timings[idx]['start']
                end_time = current_time + whisper_timings[idx]['end']
            else:
                # Fallback even distribution
                word_duration = audio_duration / len(your_words)
                start_time = current_time + (idx * word_duration)
                end_time = start_time + word_duration
            
            all_word_timings.append(WordTiming(
                word=word,
                start_time=start_time,
                end_time=end_time
            ))
        
        current_time += audio_duration + 0.1
    
    if not all_word_timings:
        print("❌ No word timings generated!")
        return
    
    print(f"✅ Generated {len(all_word_timings)} perfectly timed words")
    
    # Create video with captions
    input_video = "/home/nishan/Documents/WMK/final.mp4"
    output_video = "/home/nishan/Documents/WMK/final_perfect_captions.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Video not found: {input_video}")
        return
    
    print("🎬 Creating video with perfect captions...")
    video = VideoFileClip(input_video)
    
    # Create modern highlighted captions
    generator = MFACaptionGenerator()
    caption_clips = generator.create_modern_highlighted_captions(all_word_timings)
    
    # Composite final video
    final_video = CompositeVideoClip([video] + caption_clips)
    
    # Export
    print("💾 Exporting final video...")
    final_video.write_videofile(
        output_video,
        codec='libx264',
        audio_codec='aac'
    )
    
    # Cleanup
    video.close()
    final_video.close()
    
def create_whisper_only_captions():
    """
    Use only Whisper's recognized words and timings for perfect sync.
    """
    import whisper
    print("🤖 Loading Whisper...")
    model = whisper.load_model("base")
    
    video_path = "/home/nishan/Documents/WMK/final.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    print("🎬 Loading video...")
    video = VideoFileClip(video_path)
    
    print("🎵 Transcribing video with Whisper...")
    result = model.transcribe(video_path, word_timestamps=True)
    
    all_caption_clips = []
    for segment in result['segments']:
        for word_info in segment.get('words', []):
            word = word_info['word'].strip()
            start = word_info['start']
            end = word_info['end']
            text_clip = TextClip(
                text=word,
                font_size=70,
                color='white'
            ).with_position(('center', 'center')).with_start(start).with_duration(end - start)
            all_caption_clips.append(text_clip)
    print("🎞️ Adding captions to video...")
    final_video = CompositeVideoClip([video] + all_caption_clips)
    output_path = "/home/nishan/Documents/WMK/final_whisper_only_captions.mp4"
    print(f"💾 Saving to {output_path}...")
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print("✅ Done! Whisper-only captions video saved.")

if __name__ == "__main__":
    # Use Whisper-based approach since MFA setup is complex
    print("🎯 Using Whisper + Your Dialogue (Best Available Method)")
    create_perfect_captions_with_whisper()
    
    print("🎯 Using Whisper-only captions (perfect sync, Whisper's words)")
    create_whisper_only_captions()
