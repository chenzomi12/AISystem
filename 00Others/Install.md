## 环境安装
机器学习系统书籍部署在GitHub是依赖于d2lbook工具实现的。因此我们首先要安装d2lbook。
```bash
git clone https://github.com/openmlsys/d2l-book.git
cd d2l-book
python setup.py install
```
使用d2lbook构建HTML需要安装`pandoc`, 可以使用`conda install pandoc` （如果是MacOS可以用Homebrew）， apt源中pandoc发布版本较低，表格转换格式可能有误，请尽量使用较新版本的pandoc。
构建PDF时如果有SVG图片需要安装LibRsvg来转换SVG图片，安装`librsvg`可以通过`apt-get install librsvg`（如果是MacOS可以用Homebrew）。
当然构建PDF必须要有LaTeX，如安装[Tex Live](https://www.tug.org/texlive/).

## 编译HTML版本
在编译前先下载[openmlsys-zh](https://github.com/openmlsys/openmlsys-zh) ， 所有的编译命令都在这个文件目录内执行。
```bash
 git clone https://github.com/openmlsys/openmlsys-zh.git
 cd openmlsys-zh
```
使用d2lbook工具编译HTML。 请尽量使用build_html.sh脚本进行编译，保证首页正确合并到书籍中去。
```
sh build_html.sh
```

生成的html会在`_build/html`。

此时我们将编译好的html整个文件夹下的内容拷贝至openmlsys.github.io的docs发布。

需要注意的是docs目录下的.nojekyll不要删除了，不然网页会没有渲染。

## 样式规范

贡献请遵照本教程的[样式规范](style.md)。

## 中英文术语对照

翻译请参照[中英文术语对照](terminology.md)。
