import re

files = {
    "main.tex": r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\main.tex",
    "old_main.tex": r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\old_main.tex",
    "supplementary.tex": r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\supplementary.tex",
    "response.tex": r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\response.tex"
}

def verify():
    errors = 0
    for name, path in files.items():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 1. Brier score for SVM (should be 0.0270, not 0.0267)
        if "0.0267" in content:
            print(f"[ERROR] {name} contains outdated SVM Brier score: 0.0267")
            errors += 1
            
        # 2. SVM F1 Macro (should be 0.9329 or 93.29%, not 0.9328 or 93.28%)
        if "0.9328" in content or "93.28" in content:
            print(f"[ERROR] {name} contains outdated SVM F1 Macro: 0.9328 or 93.28%")
            errors += 1

        # 3. SOTA Proposed accuracy (should be 96.16% or 96.16\%, not 96.19% or 96.19\%)
        # Note: 96.19 is allowed for class imbalance baseline, so we only check the SOTA metric section or study row.
        # Let's search for "This Study" row in SOTA table
        if name in ["main.tex", "old_main.tex"]:
            for line in content.splitlines():
                if "This Study" in line and "96.19" in line:
                    print(f"[ERROR] {name} SOTA table row contains outdated proposed metric: 96.19%")
                    errors += 1
                if "TF-IDF+LR classifier achieved a test accuracy of $96.19" in line:
                    print(f"[ERROR] {name} SOTA explanation contains outdated proposed metric: 96.19%")
                    errors += 1
                if "TF-IDF+LR classifier achieves a test accuracy of $96.19" in line:
                    print(f"[ERROR] {name} SOTA explanation contains outdated proposed metric: 96.19%")
                    errors += 1
                    
        # 4. Logistic Regression CV Acc (should be 95.30 or 0.9530, not 95.29 or 0.9529)
        if "95.29" in content or "0.9529" in content:
            print(f"[ERROR] {name} contains outdated LR CV Acc: 95.29 / 0.9529")
            errors += 1
            
        # 5. TF-IDF+LR CV F1-Macro (should be 92.70 or 0.9270, not 92.68 or 0.9268)
        if "92.68" in content or "0.9268" in content:
            print(f"[ERROR] {name} contains outdated TF-IDF+LR CV F1: 92.68 / 0.9268")
            errors += 1

        # 6. SOTA proposed metric in response.tex (should be 96.16%, not 96.19%)
        if name == "response.tex":
            if "96.19%" in content and "near-transformer-level" in content:
                print(f"[ERROR] response.tex contains outdated SOTA proposed metric: 96.19%")
                errors += 1

    if errors == 0:
        print("[SUCCESS] All files verified. No outdated metrics found!")
    else:
        print(f"[FAILURE] Found {errors} errors.")

if __name__ == "__main__":
    verify()
