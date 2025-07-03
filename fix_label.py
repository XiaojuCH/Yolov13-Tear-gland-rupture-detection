# bulk_fix_labels.py
import glob
import os

# 你的标签目录
label_dirs = [
    r"D:\Projects_\Tears_Check_YOLO12\dataset_all/train/labels",
    r"D:\Projects_\Tears_Check_YOLO12\dataset_all/val/labels",
    r"D:\Projects_\Tears_Check_YOLO12\dataset_all/test/labels",
]

for d in label_dirs:
    for path in glob.glob(os.path.join(d, "*.txt")):
        lines = []
        changed = False
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # 如果 class_id != '0'，就改成 '0'
                if parts[0] != '0':
                    parts[0] = '0'
                    changed = True
                lines.append(" ".join(parts) + "\n")
        if changed:
            # 备份原文件
            bak = path + ".bak"
            if not os.path.exists(bak):
                os.rename(path, bak)
            # 写回修正后的标签
            with open(path, 'w') as f:
                f.writelines(lines)
            print(f"Fixed {path} (backup at {os.path.basename(bak)})")

print("✅ 所有标签文件的 class_id 已批量改为 0")
