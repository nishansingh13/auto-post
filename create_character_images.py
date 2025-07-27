#!/usr/bin/env python3
"""
Create simple character placeholder images for Peter and Stewie
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_character_placeholder(name, color, filename):
    # Create a 200x200 image with transparent background
    img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple circle for the character
    draw.ellipse([50, 50, 150, 150], fill=color, outline='black', width=3)
    
    # Try to add text (fallback to default font if custom font not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw the character name
    text_x = (200 - text_width) // 2
    text_y = (200 - text_height) // 2
    draw.text((text_x, text_y), name, fill='white', font=font)
    
    # Save the image
    img.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    # Create Peter (orange/pink color like his shirt)
    create_character_placeholder("PETER", (255, 165, 0, 255), "peter.png")
    
    # Create Stewie (yellow/white color)  
    create_character_placeholder("STEWIE", (255, 255, 100, 255), "stewie.png")
    
    print("Character placeholder images created!")
    print("You can replace these with actual character images later.")
