import sys
import os
from PIL import Image, ImageFilter

def prepare_image(img):
    """Transform image to greyscale and blur it"""
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    if img.mode != 'L':
        img = img.convert('L')
    return img

def remove_noise(img, pass_factor):
    for column in range(img.size[0]):
        for line in range(img.size[1]):
            value = remove_noise_by_pixel(img, column, line, pass_factor)
            img.putpixel((column, line), value)
    return img

def remove_noise_by_pixel(img, column, line, pass_factor):
    if img.getpixel((column, line)) < pass_factor:
        return 0
    return 255

def process_directory(directory, pass_factor):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            img = Image.open(file_path)
            img = prepare_image(img)
            img = remove_noise(img, pass_factor)
            img.save(file_path)

if __name__ == "__main__":
    input_directory = "mix_folder"
    pass_factor = 142

    process_directory(input_directory, pass_factor)
