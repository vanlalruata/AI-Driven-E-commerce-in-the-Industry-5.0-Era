import re
import difflib

main_revision_path = r"manuscript/main_revision.tex"
main_path = r"manuscript/main.tex"

with open(main_revision_path, "r", encoding="utf-8") as f:
    rev = f.read()

with open(main_path, "r", encoding="utf-8") as f:
    clean = f.read()

def strip_updates(text):
    while True:
        match = re.search(r'\\update\{', text)
        if not match:
            break
        start_idx = match.start()
        # Find matching closing brace
        brace_count = 1
        j = match.end()
        while j < len(text) and brace_count > 0:
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
            j += 1
        if brace_count == 0:
            content = text[match.end() : j-1]
            text = text[:start_idx] + content + text[j:]
        else:
            break
    return text

stripped = strip_updates(rev)
diff = list(difflib.unified_diff(
    stripped.splitlines(keepends=True),
    clean.splitlines(keepends=True),
    fromfile='stripped_revision',
    tofile='clean_main'
))

print("=== DIFF BETWEEN STRIPPED REVISION AND CLEAN MAIN ===")
print("".join(diff))
