from PIL import Image, ImageDraw, ImageFont
import random
import os
import csv


with open("words.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()


output_dir = "./dataset_pil"
os.makedirs(output_dir, exist_ok=True)


csv_file = os.path.join(output_dir, "labels.csv")


with open(csv_file, mode="w", newline="", encoding="utf-8") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["filename", "label"])  

    
    for i in range(80000):  
        text = random.choice(words)

        
        font_size = random.randint(30, 64)
        font = ImageFont.truetype("arial.ttf", font_size)

        
        img = Image.new("RGB", (300, 80), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        
        draw.text((10, 10), text, font=font, fill=(0, 0, 0))

        
        filename = f"img_{i}.png"
        filepath = os.path.join(output_dir, filename)

        
        img.save(filepath)

        
        writer.writerow([filename, text])
