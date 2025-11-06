# 读取并替换
with open(r'c:\Users\Administrator\ble_smartlit\streamlit_app_01.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换
content = content.replace("use_container_width=True", "width='stretch'")

# 写回
with open(r'c:\Users\Administrator\ble_smartlit\streamlit_app_01.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
