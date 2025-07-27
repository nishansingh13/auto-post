from moviepy import *
import os


# Configuration
output_dir = "output"
audio_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".wav")])
minecraft_video = "minecraft.mp4"  # Replace with actual Minecraft video filename
final_output = "final_video.mp4"

# Character images (add these files to your project)
peter_image = "peter_griffin.png"  # Path to Peter image
stewie_image = "stewie.png"  # Path to Stewie image

# Dialogue for captions
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

# Load Minecraft gameplay video and crop to 9:16 aspect ratio
background = VideoFileClip(minecraft_video)

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

# Create a modern TikTok/CapCut style caption
# Create a modern TikTok/CapCut style caption
# Create a modern TikTok/CapCut style caption
# Create a modern TikTok/CapCut style caption
def make_caption(text, duration, speaker):
    # Add padding by using a larger area but centering the text
    padded_text = f"    {text}    "  # Add spaces for left/right padding
    
    caption = (
        TextClip(text=padded_text, 
                font_size=58,  
                color='black',  
                stroke_color='white',  
                stroke_width=6,  # Reduced stroke width
                method='caption',  
                size=(900, 400))  # Added height=400 instead of None - this prevents bottom cutting!
        .with_duration(duration)
        .with_position(('center', 160))  
    )
    
    return caption

# Create character image overlay
def create_character_overlay(speaker, duration):
    if speaker == "peter":
        img_path = peter_image
        position = (30, 'bottom')  # Bottom left for Peter with small margin
    else:  # stewie
        img_path = stewie_image
        position = ('right', 'bottom')  # Bottom right for Stewie
    
    if os.path.exists(img_path):
        return (ImageClip(img_path)
                .resized(height=700)  # Much larger images (increased from 500 to 700)
                .with_duration(duration)
                .with_position(position)
                .with_opacity(1.0))  # Full opacity for better visibility
    else:
        print(f"Warning: {img_path} not found!")
        return None


# Process each line of dialogue
for i, (filename, (speaker, line)) in enumerate(zip(audio_files, dialogue)):
    audio_path = os.path.join(output_dir, filename)
    audio_clip = AudioFileClip(audio_path)
    
    # Create modern CapCut-style caption (only showing the current speaker's name)
    text_clip = make_caption(line, audio_clip.duration, speaker)
    
    # Create character overlay (only shows when that character is speaking)
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
