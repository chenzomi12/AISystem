import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

watermark_path = "/Users/a1-6/Workspaces/AISystem/images/watermark.png" # 水印图片的路径
source_folder = "/Users/a1-6/Workspaces/AISystem/01Introduction/images" # 原始图片的文件夹
output_folder = "/Users/a1-6/Workspaces/AISystem/01Introduction/watermark" # 输出图片的文件夹

watermark = Image.open(watermark_path).convert("RGBA") # 打开并转换水印图片
watermark_width, watermark_height = watermark.size # 获取水印图片的尺寸
w_ration = watermark_width/watermark_height

def check_image(img_path):
    if(img_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    else:
        return False

for filename in os.listdir(source_folder): # 遍历原始图片的文件夹
    if check_image(filename): # 判断是否是图片文件
        
        print("dealing with images:" + filename)
        image_path = os.path.join(source_folder, filename) # 拼接图片文件的路径
        image = Image.open(image_path).convert("RGBA") # 打开并转换图片文件

        margin = 10 # 边距
        image_width, image_height = image.size # 获取图片文件的尺寸
        new_watermark_hight = int(image_height/10)
        new_watermark_width = int(image_height/10 * w_ration)
        watermark_x = image_width - new_watermark_width - margin # 水印图片在 x 轴上的位置
        watermark_y = image_height - new_watermark_hight - margin # 水印图片在 y 轴上的位置

        new_watermark = watermark.resize((new_watermark_width, new_watermark_hight))
        # new_watermark = watermark.thumbnail((400, 400))
        image.paste(new_watermark, (watermark_x, watermark_y), new_watermark) # 将水印图片合成到原始图片上
        output_path = os.path.join(output_folder, filename) # 拼接输出文件的路径
        image.save(output_path, quality=100) # 保存输出文件