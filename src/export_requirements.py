import pkg_resources

# 输出文件
output_file = "requirements.txt"

with open(output_file, "w") as f:
    for dist in pkg_resources.working_set:
        # 忽略本地 file:// 路径
        if "@" not in dist.location:
            line = f"{dist.project_name}=={dist.version}"
            f.write(line + "\n")

print(f"requirements.txt 已生成，共 {len(pkg_resources.working_set)} 个包。")
