with open("manuscript/main.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "\uFFFD" in line:
        print(f"Line {idx+1}: {repr(line)}")
