import os
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

watermark_path = "/Users/a1-6/Workspaces/AISystem/images/watermark.png" # 水印图片的路径
source_folder = "/Users/a1-6/Workspaces/AISystem/images" # 原始图片的文件夹
output_folder = "/Users/a1-6/Workspaces/AISystem/watermark" # 输出图片的文件夹

watermark = Image.open(watermark_path).convert("RGBA") # 打开并转换水印图片
watermark_width, watermark_height = watermark.size # 获取水印图片的尺寸

def check_image(img_path):
    if(img_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    else:
        return False

def del_dir_byname(path):
	if os.path.exists(path):
		shutil.rmtree(path)
		print("文件夹已删除！", path)
	else:
		print("文件夹不存在！", path)


def create_dir(path):
	del_dir_byname(path)
	os.makedirs(path)
	return path

create_dir(output_folder)
for filename in os.listdir(source_folder): # 遍历原始图片的文件夹
    if check_image(filename): # 判断是否是图片文件
        
        print("dealing with images:" + filename)
        image_path = os.path.join(source_folder, filename) # 拼接图片文件的路径
        image = Image.open(image_path).convert("RGBA") # 打开并转换图片文件
        resized = image

        margin = 10 # 边距
        baise_width = 1080
        src_width, src_height = image.size # 获取图片文件的尺寸
        resized_width, resized_height = image.size # 获取图片文件的尺寸

        # resize iamges
        if src_width > baise_width:
            resized_width = baise_width
            resized_height = int(baise_width/src_width * src_height)
            resized = image.resize((resized_width, resized_height))

        # watermark images
        src_width, src_height = resized_width, resized_height
        w_ration = watermark_height/watermark_width
        new_watermark_width = int(src_width/5)
        new_watermark_hight = int(src_width/5 * w_ration)
        watermark_x = src_width - new_watermark_width - margin # 水印图片在 x 轴上的位置
        watermark_y = src_height - new_watermark_hight - margin # 水印图片在 y 轴上的位置
        # new_watermark = watermark.thumbnail((400, 400))
        new_watermark = watermark.resize((new_watermark_width, new_watermark_hight))
        resized.paste(new_watermark, (watermark_x, watermark_y), new_watermark) # 将水印图片合成到原始图片上

        # output images
        out_name = filename.split(".")[0] + ".png"
        print("Outputing images name:", out_name)
        output_path = os.path.join(output_folder, out_name) # 拼接输出文件的路径
        resized.save(output_path, quality=100) # 保存输出文件
        # break
    else:
        print("CANNOT dealing images:" + filename)