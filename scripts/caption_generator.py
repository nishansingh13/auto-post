#!/usr/bin/env python3
"""
Advanced Video Caption Generator with Word-Level Highlighting
Supports multiple approaches for generating captions with synchronized word highlighting
"""

import os
import json
import subprocess
import wave
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import check_for_imagemagick
import speech_recognition as sr
from pydub import AudioSegment
import whisper
import torch
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import warnings
warnings.filterwarnings("ignore")

@dataclass
class WordTiming:
    """Represents a word with its timing information"""
    word: str
    start_time: float
    end_time: float
    confidence: float = 0.0

class CaptionGenerator:
    def __init__(self):
        """Initialize the caption generator with multiple ASR options"""
        self.whisper_model = None
        self.wav2vec_processor = None
        self.wav2vec_model = None
        
    def setup_whisper(self, model_size: str = "base"):
        """Setup Whisper model for transcription"""
        print(f"Loading Whisper {model_size} model...")
        self.whisper_model = whisper.load_model(model_size)
        
    def setup_wav2vec(self):
        """Setup Wav2Vec2 model for word-level alignment"""
        print("Loading Wav2Vec2 model...")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # APPROACH 1: Whisper with Word-Level Timestamps
    def transcribe_with_whisper_word_timestamps(self, audio_path: str) -> List[WordTiming]:
        """
        Use Whisper to get word-level timestamps
        Most accurate for transcription but limited word-level timing precision
        """
        if not self.whisper_model:
            self.setup_whisper()
            
        print("Transcribing with Whisper (word-level timestamps)...")
        result = self.whisper_model.transcribe(
            audio_path, 
            word_timestamps=True,
            verbose=False
        )
        
        word_timings = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_timings.append(WordTiming(
                    word=word_info["word"].strip(),
                    start_time=word_info["start"],
                    end_time=word_info["end"],
                    confidence=segment.get("avg_logprob", 0.0)
                ))
        
        return word_timings

    # APPROACH 2: Forced Alignment with Wav2Vec2
    def force_align_with_wav2vec(self, audio_path: str, transcript: str) -> List[WordTiming]:
        """
        Use Wav2Vec2 for forced alignment - most precise word timing
        Requires known transcript but gives best synchronization
        """
        if not self.wav2vec_model:
            self.setup_wav2vec()
            
        print("Performing forced alignment with Wav2Vec2...")
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Tokenize transcript
        words = transcript.split()
        inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.wav2vec_model(inputs.input_values).logits
            
        # Get predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
        
        # Simple alignment heuristic (for production, use specialized alignment tools)
        word_timings = self._align_words_simple(words, transcription, audio.shape[0] / 16000)
        
        return word_timings
    
    def _align_words_simple(self, words: List[str], transcription: str, duration: float) -> List[WordTiming]:
        """Simple word alignment heuristic"""
        word_timings = []
        time_per_char = duration / len(transcription) if transcription else 0
        current_time = 0
        
        for word in words:
            word_duration = len(word) * time_per_char * 1.2  # Add buffer
            word_timings.append(WordTiming(
                word=word,
                start_time=current_time,
                end_time=current_time + word_duration,
                confidence=0.8
            ))
            current_time += word_duration + (time_per_char * 0.5)  # Small pause between words
            
        return word_timings

    # APPROACH 3: Montreal Forced Alignment (MFA) - Most Professional
    def align_with_mfa(self, audio_path: str, transcript: str) -> List[WordTiming]:
        """
        Use Montreal Forced Alignment for professional-grade word alignment
        Requires MFA installation but gives broadcast-quality results
        """
        print("Using Montreal Forced Alignment...")
        
        # Prepare files for MFA
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        temp_dir = "/tmp/mfa_align"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create transcript file
        transcript_path = os.path.join(temp_dir, f"{base_name}.txt")
        with open(transcript_path, 'w') as f:
            f.write(transcript)
        
        # Copy audio file
        audio_dest = os.path.join(temp_dir, f"{base_name}.wav")
        subprocess.run(['cp', audio_path, audio_dest])
        
        try:
            # Run MFA alignment
            cmd = [
                'mfa', 'align', 
                temp_dir, 
                'english_us_arpa', 
                'english_us_arpa', 
                os.path.join(temp_dir, 'output'),
                '--clean'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Parse TextGrid output
            textgrid_path = os.path.join(temp_dir, 'output', f"{base_name}.TextGrid")
            return self._parse_textgrid(textgrid_path)
            
        except subprocess.CalledProcessError as e:
            print(f"MFA alignment failed: {e}")
            return []
    
    def _parse_textgrid(self, textgrid_path: str) -> List[WordTiming]:
        """Parse TextGrid file from MFA"""
        word_timings = []
        
        try:
            import textgrid
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            for tier in tg:
                if tier.name == "words":
                    for interval in tier:
                        if interval.mark.strip():  # Skip empty intervals
                            word_timings.append(WordTiming(
                                word=interval.mark,
                                start_time=interval.minTime,
                                end_time=interval.maxTime,
                                confidence=1.0
                            ))
            
        except ImportError:
            print("TextGrid library not installed. Install with: pip install textgrid")
        except Exception as e:
            print(f"Error parsing TextGrid: {e}")
            
        return word_timings

    # APPROACH 4: Gentle Forced Aligner (Docker-based)
    def align_with_gentle(self, audio_path: str, transcript: str) -> List[WordTiming]:
        """
        Use Gentle forced aligner via Docker
        Good balance of accuracy and ease of use
        """
        print("Using Gentle forced aligner...")
        
        try:
            # Run Gentle in Docker container
            cmd = [
                'docker', 'run', '--rm',
                '-v', f"{os.path.dirname(audio_path)}:/gentle/audio",
                'lowerquality/gentle',
                f'/gentle/audio/{os.path.basename(audio_path)}',
                transcript
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            alignment_data = json.loads(result.stdout)
            
            word_timings = []
            for word_data in alignment_data.get('words', []):
                if word_data.get('case') == 'success':
                    word_timings.append(WordTiming(
                        word=word_data['alignedWord'],
                        start_time=word_data['start'],
                        end_time=word_data['end'],
                        confidence=word_data.get('confidence', 0.8)
                    ))
            
            return word_timings
            
        except subprocess.CalledProcessError as e:
            print(f"Gentle alignment failed: {e}")
            return []

    # APPROACH 5: Audio Feature-Based Alignment
    def align_with_audio_features(self, audio_path: str, transcript: str) -> List[WordTiming]:
        """
        Use audio features (energy, spectral features) to estimate word boundaries
        Fallback method when other aligners aren't available
        """
        print("Using audio feature-based alignment...")
        
        # Load audio
        audio, sr = librosa.load(audio_path)
        
        # Extract features
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Detect speech regions using energy thresholding
        energy_threshold = np.percentile(energy, 20)
        speech_regions = self._detect_speech_regions(energy, energy_threshold, sr)
        
        # Distribute words across speech regions
        words = transcript.split()
        word_timings = self._distribute_words_to_regions(words, speech_regions)
        
        return word_timings
    
    def _detect_speech_regions(self, energy: np.ndarray, threshold: float, sr: int) -> List[Tuple[float, float]]:
        """Detect continuous speech regions based on energy"""
        hop_length = 512
        time_frames = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
        
        speech_mask = energy > threshold
        regions = []
        start_time = None
        
        for i, (is_speech, time) in enumerate(zip(speech_mask, time_frames)):
            if is_speech and start_time is None:
                start_time = time
            elif not is_speech and start_time is not None:
                regions.append((start_time, time))
                start_time = None
        
        # Handle case where speech continues to end
        if start_time is not None:
            regions.append((start_time, time_frames[-1]))
            
        return regions
    
    def _distribute_words_to_regions(self, words: List[str], regions: List[Tuple[float, float]]) -> List[WordTiming]:
        """Distribute words evenly across detected speech regions"""
        word_timings = []
        
        if not regions:
            # Fallback: distribute evenly across entire duration
            total_duration = 10.0  # Default assumption
            time_per_word = total_duration / len(words)
            
            for i, word in enumerate(words):
                start_time = i * time_per_word
                end_time = start_time + time_per_word * 0.8  # Leave small gaps
                word_timings.append(WordTiming(word, start_time, end_time, 0.5))
            
            return word_timings
        
        # Distribute words across regions based on region duration
        total_region_duration = sum(end - start for start, end in regions)
        words_per_second = len(words) / total_region_duration
        
        word_idx = 0
        for region_start, region_end in regions:
            region_duration = region_end - region_start
            words_in_region = max(1, int(region_duration * words_per_second))
            
            # Don't exceed available words
            words_in_region = min(words_in_region, len(words) - word_idx)
            
            if words_in_region == 0:
                break
                
            time_per_word = region_duration / words_in_region
            
            for i in range(words_in_region):
                if word_idx >= len(words):
                    break
                    
                start_time = region_start + i * time_per_word
                end_time = start_time + time_per_word * 0.8
                
                word_timings.append(WordTiming(
                    word=words[word_idx],
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.6
                ))
                word_idx += 1
        
        return word_timings

    # Caption Generation Methods
    def create_highlighted_captions(self, word_timings: List[WordTiming], video_duration: float, 
                                  style: str = "modern") -> List[TextClip]:
        """
        Create video captions with word-level highlighting
        
        Args:
            word_timings: List of word timing information
            video_duration: Total video duration
            style: Caption style ("modern", "classic", "karaoke", "subtitle")
        """
        
        if style == "modern":
            return self._create_modern_captions(word_timings, video_duration)
        elif style == "karaoke":
            return self._create_karaoke_captions(word_timings, video_duration)
        elif style == "subtitle":
            return self._create_subtitle_captions(word_timings, video_duration)
        else:
            return self._create_classic_captions(word_timings, video_duration)
    
    def _create_modern_captions(self, word_timings: List[WordTiming], video_duration: float) -> List[TextClip]:
        """Create modern TikTok/Instagram-style captions with highlighting"""
        clips = []
        
        # Group words into lines (max 4-5 words per line)
        lines = self._group_words_into_lines(word_timings, max_words_per_line=4)
        
        for line_words in lines:
            if not line_words:
                continue
                
            line_start = line_words[0].start_time
            line_end = line_words[-1].end_time
            line_duration = line_end - line_start
            
            # Create base text for the line
            line_text = " ".join(word.word for word in line_words)
            
            # Create multiple clips: one for each word highlight state
            for highlight_idx, current_word in enumerate(line_words):
                # Build text with current word highlighted
                highlighted_text = ""
                for i, word in enumerate(line_words):
                    if i == highlight_idx:
                        highlighted_text += f"<span fgcolor='red' weight='bold'>{word.word}</span> "
                    else:
                        highlighted_text += f"<span fgcolor='white'>{word.word}</span> "
                
                # Create clip for this highlight state
                clip = TextClip(
                    highlighted_text.strip(),
                    font='Arial-Bold',
                    fontsize=50,
                    color='white',
                    stroke_color='black',
                    stroke_width=3,
                    method='pango'
                ).with_position(('center', 'bottom')).with_start(current_word.start_time).with_duration(
                    current_word.end_time - current_word.start_time
                )
                
                clips.append(clip)
        
        return clips
    
    def _create_karaoke_captions(self, word_timings: List[WordTiming], video_duration: float) -> List[TextClip]:
        """Create karaoke-style captions with progressive highlighting"""
        clips = []
        
        # Create full text first
        full_text = " ".join(word.word for word in word_timings)
        
        # Create clips for each word highlight
        for i, current_word in enumerate(word_timings):
            # Create text with words up to current highlighted in red, rest in white
            karaoke_text = ""
            for j, word in enumerate(word_timings):
                if j < i:
                    karaoke_text += f"<span fgcolor='red'>{word.word}</span> "
                elif j == i:
                    karaoke_text += f"<span fgcolor='yellow' weight='bold'>{word.word}</span> "
                else:
                    karaoke_text += f"<span fgcolor='white'>{word.word}</span> "
            
            clip = TextClip(
                karaoke_text.strip(),
                font='Arial-Bold',
                fontsize=40,
                color='white',
                stroke_color='black',
                stroke_width=2,
                method='pango'
            ).with_position(('center', 'bottom')).with_start(current_word.start_time).with_duration(
                current_word.end_time - current_word.start_time
            )
            
            clips.append(clip)
        
        return clips
    
    def _create_subtitle_captions(self, word_timings: List[WordTiming], video_duration: float) -> List[TextClip]:
        """Create traditional subtitle-style captions with word highlighting"""
        clips = []
        
        # Group into subtitle chunks (by pauses or max duration)
        subtitle_groups = self._group_words_into_subtitles(word_timings)
        
        for group in subtitle_groups:
            group_start = group[0].start_time
            group_end = group[-1].end_time
            
            # Create highlighted versions for each word in group
            for highlight_word in group:
                subtitle_text = ""
                for word in group:
                    if word == highlight_word:
                        subtitle_text += f"<u><b>{word.word}</b></u> "
                    else:
                        subtitle_text += f"{word.word} "
                
                clip = TextClip(
                    subtitle_text.strip(),
                    font='Arial',
                    fontsize=36,
                    color='white',
                    bg_color='black',
                    method='pango'
                ).with_position(('center', 0.8)).with_start(highlight_word.start_time).with_duration(
                    highlight_word.end_time - highlight_word.start_time
                )
                
                clips.append(clip)
        
        return clips
    
    def _create_classic_captions(self, word_timings: List[WordTiming], video_duration: float) -> List[TextClip]:
        """Create classic-style captions with simple highlighting"""
        clips = []
        
        for word in word_timings:
            # Simple approach: each word appears individually
            clip = TextClip(
                word.word,
                font='Arial-Bold',
                fontsize=60,
                color='yellow',
                stroke_color='black',
                stroke_width=3
            ).with_position(('center', 'center')).with_start(word.start_time).with_duration(
                word.end_time - word.start_time
            )
            
            clips.append(clip)
        
        return clips
    
    def _group_words_into_lines(self, word_timings: List[WordTiming], max_words_per_line: int = 4) -> List[List[WordTiming]]:
        """Group words into lines for display"""
        lines = []
        current_line = []
        
        for word in word_timings:
            current_line.append(word)
            
            # Start new line if max words reached or if there's a significant pause
            if len(current_line) >= max_words_per_line:
                lines.append(current_line)
                current_line = []
            elif (len(current_line) > 1 and 
                  word.start_time - current_line[-2].end_time > 0.5):  # 500ms pause
                lines.append(current_line)
                current_line = []
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _group_words_into_subtitles(self, word_timings: List[WordTiming], max_duration: float = 3.0) -> List[List[WordTiming]]:
        """Group words into subtitle chunks"""
        groups = []
        current_group = []
        group_start = None
        
        for word in word_timings:
            if not current_group:
                group_start = word.start_time
            
            current_group.append(word)
            
            # Create new group if max duration exceeded or significant pause
            if (word.end_time - group_start > max_duration or
                (len(current_group) > 1 and word.start_time - current_group[-2].end_time > 1.0)):
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups

    # Main processing function
    def generate_captions_for_video(self, 
                                  video_path: str, 
                                  output_path: str,
                                  method: str = "whisper",
                                  transcript: str = None,
                                  style: str = "modern") -> str:
        """
        Main function to generate captions with word highlighting
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            method: Alignment method ("whisper", "wav2vec", "mfa", "gentle", "audio_features")
            transcript: Known transcript (required for some methods)
            style: Caption style ("modern", "karaoke", "subtitle", "classic")
        
        Returns:
            Path to output video with captions
        """
        
        print(f"Generating captions for {video_path} using {method} method...")
        
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio_path = "/tmp/temp_audio.wav"
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        # Get word timings based on selected method
        if method == "whisper":
            word_timings = self.transcribe_with_whisper_word_timestamps(audio_path)
        elif method == "wav2vec" and transcript:
            word_timings = self.force_align_with_wav2vec(audio_path, transcript)
        elif method == "mfa" and transcript:
            word_timings = self.align_with_mfa(audio_path, transcript)
        elif method == "gentle" and transcript:
            word_timings = self.align_with_gentle(audio_path, transcript)
        elif method == "audio_features" and transcript:
            word_timings = self.align_with_audio_features(audio_path, transcript)
        else:
            raise ValueError(f"Invalid method '{method}' or missing transcript")
        
        if not word_timings:
            raise ValueError("No word timings generated")
        
        print(f"Generated {len(word_timings)} word timings")
        
        # Create caption clips
        caption_clips = self.create_highlighted_captions(word_timings, video.duration, style)
        
        # Composite video with captions
        final_video = CompositeVideoClip([video] + caption_clips)
        
        # Write output
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Cleanup
        os.remove(audio_path)
        video.close()
        final_video.close()
        
        print(f"Caption video saved to: {output_path}")
        return output_path

# Usage Examples and Utility Functions
def install_dependencies():
    """Install required dependencies"""
    packages = [
        "whisper-openai",
        "torch",
        "transformers",
        "librosa",
        "soundfile", 
        "pydub",
        "moviepy",
        "speechrecognition",
        "pillow",
        "matplotlib",
        "numpy"
    ]
    
    for package in packages:
        subprocess.run([f"pip install {package}"], shell=True)

if __name__ == "__main__":
    # Example usage
    generator = CaptionGenerator()
    
    # Example 1: Using Whisper (easiest, no transcript needed)
    video_path = "../minecraft_video.mp4"
    if os.path.exists(video_path):
        output_path = "../minecraft_with_captions_whisper.mp4"
        generator.generate_captions_for_video(
            video_path=video_path,
            output_path=output_path,
            method="whisper",
            style="modern"
        )
        print(f"Generated captions using Whisper: {output_path}")
    
    # Example 2: Using forced alignment with known transcript (most accurate)
    transcript = "Your known transcript here..."
    if os.path.exists(video_path) and transcript.strip():
        output_path = "../minecraft_with_captions_aligned.mp4"
        generator.generate_captions_for_video(
            video_path=video_path,
            output_path=output_path,
            method="wav2vec",
            transcript=transcript,
            style="karaoke"
        )
        print(f"Generated aligned captions: {output_path}")