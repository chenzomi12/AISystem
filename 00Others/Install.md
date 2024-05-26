## 环境安装
机器学习系统书籍部署在 GitHub 是依赖于 d2lbook 工具实现的。因此我们首先要安装 d2lbook。
```bash
git clone https://github.com/openmlsys/d2l-book.git
cd d2l-book
python setup.py install
```
使用 d2lbook 构建 HTML 需要安装`pandoc`, 可以使用`conda install pandoc` （如果是 MacOS 可以用 Homebrew）， apt 源中 pandoc 发布版本较低，表格转换格式可能有误，请尽量使用较新版本的 pandoc。
构建 PDF 时如果有 SVG 图片需要安装 LibRsvg 来转换 SVG 图片，安装`librsvg`可以通过`apt-get install librsvg`（如果是 MacOS 可以用 Homebrew）。
当然构建 PDF 必须要有 LaTeX，如安装[Tex Live](https://www.tug.org/texlive/).

## 编译 HTML 版本
在编译前先下载[openmlsys-zh](https://github.com/openmlsys/openmlsys-zh) ， 所有的编译命令都在这个文件目录内执行。
```bash
 git clone https://github.com/openmlsys/openmlsys-zh.git
 cd openmlsys-zh
```
使用 d2lbook 工具编译 HTML。 请尽量使用 build_html.sh 脚本进行编译，保证首页正确合并到书籍中去。
```
sh build_html.sh
```

生成的 html 会在`_build/html`。

此时我们将编译好的 html 整个文件夹下的内容拷贝至 openmlsys.github.io 的 docs 发布。

需要注意的是 docs 目录下的.nojekyll 不要删除了，不然网页会没有渲染。

## 样式规范

贡献请遵照本教程的[样式规范](style.md)。

## 中英文术语对照

翻译请参照[中英文术语对照](terminology.md)。
