# 适用于[License] (https://github.com/chenzomi12/AISystem/blob/main/LICENSE)版权许可

from bs4 import BeautifulSoup
import os

"""
1. 本脚本文件会设置所有html文件，使得表格居中显示。
2. 确保你已经安装了bs4, 具体可以通过pip install bs4 进行安装
"""

root_path = "./"
html_root_path = "_build/html/"


def get_html_list():
    index_html_path = os.path.join(root_path, html_root_path, "index.html")
    index_soup = BeautifulSoup(open(index_html_path))

    content_list = index_soup.find(name="div", attrs={"class": "globaltoc"}). \
        find_all(name="a", attrs={"class": "reference internal"})
    html_list = [os.path.join(html_root_path, content_name["href"]) for content_name in content_list]
    return html_list


def format_table():
    html_list = get_html_list()
    for html_file in html_list:
        try:
            soup = BeautifulSoup(open(html_file))
            all_tables = soup.find_all(name="table", attrs={"class": "docutils align-default"})
            for table in all_tables:
                table["style"] = "margin-left:auto;margin-right:auto;margin-top:10px;margin-bottom:20px;"

            if len(all_tables):
                write_out_file = open(html_file, mode="w")
                write_out_file.write(soup.prettify())
                write_out_file.close()
        except:
            pass


if __name__ == "__main__":
    format_table()