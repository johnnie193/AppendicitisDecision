import os

def get_subdirectories(folder_path):
    # 使用列表推导式获取所有子文件夹名称
    return [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# 示例：指定文件夹路径
folder_path = "C:\\Users\\xuty0\\Desktop\\research\\lwy\\mark\\negative"
subdirectories = get_subdirectories(folder_path)
print(subdirectories)

folder_path = "C:\\Users\\xuty0\\Desktop\\research\\lwy\\mark\\positive"
subdirectories = get_subdirectories(folder_path)
print(subdirectories)