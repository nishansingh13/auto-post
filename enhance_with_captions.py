#!/usr/bin/env python3
"""
Simple Example: Add word-highlighting captions to your existing Peter/Stewie video
Uses the dialogue from your existing project
"""

import os
import sys
sys.path.append('/home/nishan/Documents/WMK/scripts')

from caption_generator import CaptionGenerator, WordTiming
from moviepy import VideoFileClip, CompositeVideoClip
import numpy as np

def create_word_timings_from_dialogue():
    """
    Create word timings based on your existing dialogue and audio files
    This uses your existing audio files to estimate word timing
    """
    
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
    
    # Get audio durations from your existing files
    from moviepy import AudioFileClip
    output_dir = "/home/nishan/Documents/WMK/output"
    
    all_word_timings = []
    current_time = 0
    
    for i, (speaker, text) in enumerate(dialogue):
        audio_file = os.path.join(output_dir, f"{i+1}.wav")
        
        if os.path.exists(audio_file):
            # Get actual audio duration
            audio_clip = AudioFileClip(audio_file)
            audio_duration = audio_clip.duration
            audio_clip.close()
            
            # Split text into words
            words = text.split()
            
            # Distribute words across the audio duration
            # Using smart timing based on word complexity
            word_timings = smart_word_distribution(words, current_time, audio_duration)
            all_word_timings.extend(word_timings)
            
            current_time += audio_duration + 0.1  # Small gap between clips
        else:
            print(f"Warning: Audio file {audio_file} not found")
    
    return all_word_timings

def smart_word_distribution(words, start_time, duration):
    """
    Intelligently distribute words across audio duration
    Similar to the advanced timing from your movie.py
    """
    def estimate_syllables(word):
        """Estimate syllables for timing"""
        word = word.lower().strip(".,!?;:'-\"")
        if len(word) <= 1:
            return 1
        
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(syllable_count, 1)
    
    # Calculate relative word weights
    word_weights = []
    for word in words:
        syllables = estimate_syllables(word)
        base_weight = syllables * 0.35
        
        # Add complexity for punctuation
        if word.endswith(('.', '!', '?')):
            base_weight += 0.4
        elif word.endswith(','):
            base_weight += 0.2
            
        word_weights.append(max(base_weight, 0.18))
    
    # Normalize to total duration
    total_weight = sum(word_weights)
    word_durations = [(w / total_weight) * duration for w in word_weights]
    
    # Create WordTiming objects
    word_timings = []
    current_time = start_time
    
    for word, word_duration in zip(words, word_durations):
        word_timings.append(WordTiming(
            word=word,
            start_time=current_time,
            end_time=current_time + word_duration,
            confidence=0.8
        ))
        current_time += word_duration
    
    return word_timings

def create_enhanced_video_with_captions():
    """
    Create your Peter/Stewie video with modern word-highlighting captions
    """
    
    print("🎬 Creating enhanced video with word-highlighting captions...")
    
    # Paths
    input_video = "/home/nishan/Documents/WMK/final.mp4"  # Your existing video
    output_video = "/home/nishan/Documents/WMK/final_with_word_captions.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        print("Please run your movie.py script first to create the base video")
        return
    
    # Initialize caption generator
    generator = CaptionGenerator()
    
    # Create word timings from your dialogue
    print("📝 Creating word timings from dialogue...")
    word_timings = create_word_timings_from_dialogue()
    
    if not word_timings:
        print("❌ No word timings created")
        return
    
    print(f"✅ Created {len(word_timings)} word timings")
    
    # Load your existing video
    video = VideoFileClip(input_video)
    
    # Create modern captions with word highlighting
    print("🎨 Creating modern captions with word highlighting...")
    caption_clips = generator.create_highlighted_captions(
        word_timings, 
        video.duration, 
        style="modern"  # Change to "karaoke", "subtitle", or "classic"
    )
    
    # Composite video with new captions
    print("🎞️ Compositing final video...")
    final_video = CompositeVideoClip([video] + caption_clips)
    
    # Write output
    print("💾 Saving enhanced video...")
    final_video.write_videofile(
        output_video,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    
    # Cleanup
    video.close()
    final_video.close()
    
    print(f"✅ Enhanced video with word captions saved: {output_video}")

def demo_different_caption_styles():
    """
    Create multiple versions with different caption styles
    """
    styles = ["modern", "karaoke", "subtitle", "classic"]
    
    for style in styles:
        print(f"\n🎨 Creating {style} style captions...")
        
        # Initialize generator
        generator = CaptionGenerator()
        word_timings = create_word_timings_from_dialogue()
        
        if not word_timings:
            continue
            
        input_video = "/home/nishan/Documents/WMK/final.mp4"
        output_video = f"/home/nishan/Documents/WMK/final_with_{style}_captions.mp4"
        
        if not os.path.exists(input_video):
            print(f"❌ Input video not found: {input_video}")
            continue
        
        video = VideoFileClip(input_video)
        caption_clips = generator.create_highlighted_captions(word_timings, video.duration, style)
        final_video = CompositeVideoClip([video] + caption_clips)
        
        final_video.write_videofile(
            output_video,
            codec='libx264', 
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        video.close()
        final_video.close()
        
        print(f"✅ {style.title()} style video saved: {output_video}")

if __name__ == "__main__":
    # Create enhanced video with word-highlighting captions
    create_enhanced_video_with_captions()
    
    # Uncomment to create all styles
    # demo_different_caption_styles()
