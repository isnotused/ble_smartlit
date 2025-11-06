#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量替换 use_container_width 为 width"""

file_path = r'c:\Users\Administrator\ble_smartlit\streamlit_app_01.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 执行替换
content = content.replace("use_container_width=True", "width='stretch'")
content = content.replace("use_container_width=False", "width='content'")

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("替换完成！")
print(f"文件: {file_path}")
