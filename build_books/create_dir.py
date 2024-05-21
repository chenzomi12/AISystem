# 适用于[License] (https://github.com/chenzomi12/AISystem/blob/main/LICENSE)版权许可

import glob

tempate = """
```toc
:maxdepth: 1

"""

for file_name in glob.glob():
    README_list = []
    tempate += README_list
    tempate += "```"
    print(tempate)


