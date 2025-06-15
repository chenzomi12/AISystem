# 适用于[License] (https://github.com/chenzomi12/AISystem/blob/main/LICENSE)版权许可

import os
import shutil


TEMP = """
```{toctree}
:maxdepth: 1

"""


def del_dir_byname(path):
	if os.path.exists(path):
		shutil.rmtree(path)
		print("文件夹已删除！", path)
	else:
		print("文件夹不存在！", path)


def create_dir(path, name):
	new_path = os.path.join(path, name)
	del_dir_byname(new_path)
	if "images" in name:
		return None

	os.makedirs(new_path)
	return new_path


def copystrtodir(path1, path2, rename):
	new_path2 = os.path.join(path2, rename)
	del_dir_byname(path2)
	del_dir_byname(new_path2)
	shutil.copytree(path1, new_path2)


def check_markdown(file_name):
	root_ext = os.path.splitext(file_name)
	if root_ext[1] == '.md':
		return True
	else:
		return False

def check_pdf(file_name):
	root_ext = os.path.splitext(file_name)
	if root_ext[1] == '.pdf':
		return True
	else:
		return False

def add2readme(file_path, string):
	if file_path.split('/')[-1] == 'README.md':
		with open(file_path, encoding="utf-8", mode="a") as file:  
			file.write(string)


def change_iamgepath_markdown(file_path):
	"""
	change ![ENIAC01](images/01CPUBase01.png)
	to ![ENIAC01](../images/02Hardware02ChipBase/01CPUBase01.png)

	"""
	search_text = "images/"
	replace_text = "../images/" + file_path.split('/')[-2] + "/"
	print(search_text,replace_text, file_path)
	with open(file_path, 'r', encoding='UTF-8') as file:
		data = file.read()
		data = data.replace(search_text, replace_text)

	with open(file_path, 'w',encoding='UTF-8') as file:
		file.write(data)


def get_subfile(path, dir_path):
	file_path = os.listdir(path)
	target_filenames = []
	target_pdf_filenames = []
	temp = TEMP
	file_path.sort()
	image_name = '/images/' + dir_path.split('/')[-1]
	save_name = dir_path.split('/')[:-1]
	save_path = '/'.join(save_name) + image_name
	
	## 找到所有的 md 并记录下来
	for file in file_path:
		fp = os.path.join(path, file)
		if os.path.isfile(fp) and check_markdown(fp):
			print("dealing with MD: ", fp)
			target_filenames.append(fp)

			if fp.split('/')[-1] == 'README.md':
				continue
			
			temp += fp.split('/')[-1][:-3]
			temp += "\n"
		
		# 移动 images 目录到外层
		elif os.path.isdir(fp) and fp.split('/')[-1] == "images":
			shutil.copytree(fp, save_path, dirs_exist_ok = True)
	temp += "```"

	## 找到所有的 pdf 并记录下来
	for file in file_path:
		fp = os.path.join(path, file)
		if os.path.isfile(fp) and check_pdf(fp):
			print("dealing with PDF: ", fp)
			target_pdf_filenames.append(fp)

	## 迁移文件到新的地方
	print("now we are going to move MD: ", target_filenames)
	for filename in target_filenames:
		shutil.copy(filename, dir_path)

	# 修改 markdown 里面的图片地址
	## 写 readme
	print("write temp to readme...")
	file_path = os.listdir(dir_path)
	for file in file_path:
		fp = os.path.join(dir_path, file)
		add2readme(fp, temp)
		change_iamgepath_markdown(fp)

	print("now we are going to move PDF: ", target_pdf_filenames)
	for filename in target_pdf_filenames:
		shutil.copy(filename, dir_path)

	return target_filenames


def getallfile(path):
	file_path = os.listdir(path)

	# 遍历该文件夹下的所有目录或者文件
	for file in file_path:
		fp = os.path.join(path, file)
		if os.path.isdir(fp):
			file_dist = fp.split('/')
			new_dir_name = ''.join(file_dist[-2:])
			new_path = create_dir(dir_paths, new_dir_name)
			if new_path:
				get_subfile(fp, new_path)

		elif os.path.isfile(fp):
			# 遍历 md 文件，并复制到指定目录
			if check_markdown(fp):
				new_dir_name = fp.split('/')[-2]
				new_path = create_dir(dir_paths, new_dir_name)
				shutil.copy(fp, new_path)


target_dir0 = '/Users/a1-6/Workspaces/AISystem/01Introduction'
target_dir1 = '/Users/a1-6/Workspaces/AISystem/02Hardware'
target_dir2 = '/Users/a1-6/Workspaces/AISystem/03Compiler'
target_dir3 = '/Users/a1-6/Workspaces/AISystem/04Inference'
target_dir4 = '/Users/a1-6/Workspaces/AISystem/05Framework'
dir_paths = '/Users/a1-6/Workspaces/AISystem_BOOK/source/'

getallfile(target_dir0)
getallfile(target_dir1)
getallfile(target_dir2)
getallfile(target_dir3)
getallfile(target_dir4)

