import re

bib_path = r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\bibliography.bib"

with open(bib_path, "r", encoding="utf-8") as f:
    bib_content = f.read()

# Simple entry splitter
entries = re.split(r'@(\w+)\s*\{', bib_content)
bib_dict = {}

i = 1
while i < len(entries):
    entry_type = entries[i].lower()
    entry_body = entries[i+1]
    i += 2
    
    # Parse key
    key_match = re.match(r'\s*([a-zA-Z0-9_\-\:]+)\s*,', entry_body)
    if not key_match:
        continue
    key = key_match.group(1)
    
    # Parse fields using brace-counting
    fields = {}
    pos = key_match.end()
    while pos < len(entry_body):
        # Find next field name
        field_name_match = re.search(r'([a-zA-Z0-9_\-]+)\s*=\s*', entry_body[pos:])
        if not field_name_match:
            break
        
        name = field_name_match.group(1).lower()
        start_val = pos + field_name_match.end()
        
        # Determine value based on starting character
        char = entry_body[start_val]
        if char == '{':
            # Count braces
            brace_count = 1
            j = start_val + 1
            while j < len(entry_body) and brace_count > 0:
                if entry_body[j] == '{':
                    brace_count += 1
                elif entry_body[j] == '}':
                    brace_count -= 1
                j += 1
            val = entry_body[start_val+1 : j-1]
            pos = j
        elif char == '"':
            # Find next non-escaped quote
            j = start_val + 1
            while j < len(entry_body):
                if entry_body[j] == '"' and entry_body[j-1] != '\\':
                    break
                j += 1
            val = entry_body[start_val+1 : j]
            pos = j + 1
        else:
            # Match until comma or newline or end
            match_end = re.search(r'[,#\n]', entry_body[start_val:])
            if match_end:
                val = entry_body[start_val : start_val + match_end.start()]
                pos = start_val + match_end.end()
            else:
                val = entry_body[start_val:]
                pos = len(entry_body)
        
        fields[name] = val.strip()
    
    bib_dict[key] = (entry_type, fields)

def clean_braces(s):
    # First, let's replace common LaTeX accent sequences with standard text or clean LaTeX equivalents
    # e.g., \'{e} -> \'e
    s = re.sub(r'\\\'\{([a-zA-Z])\}', r"\\'\1", s)
    s = re.sub(r'\\\"\{([a-zA-Z])\}', r'\\"\1', s)
    s = re.sub(r'\\`\{([a-zA-Z])\}', r'\\`\1', s)
    s = re.sub(r'\\\^\{([a-zA-Z])\}', r'\\\^\1', s)
    s = re.sub(r'\\~\{([a-zA-Z])\}', r'\\~\1', s)
    s = re.sub(r'\\c\{([a-zA-Z])\}', r'\\c \1', s)
    
    # Remove remaining braces, but keep text inside
    s = re.sub(r'\{([^}]+)\}', r'\1', s)
    s = s.replace('{', '').replace('}', '')
    s = s.replace('\\_', '_')
    s = s.replace('\\&', '&')
    return s

def clean_latex_accents_for_short_author(s):
    # For short author label, we want plain text or standard LaTeX format without weird braces
    s = s.replace("\\'{e}", "e").replace("\\'e", "e")
    s = s.replace("\\'{o}", "o").replace("\\'o", "o")
    s = s.replace("\\'{a}", "a").replace("\\'a", "a")
    s = s.replace("\\'{\\i}", "i").replace("\\'i", "i")
    s = s.replace("\\`{e}", "e").replace("\\`e", "e")
    s = s.replace("\\`{o}", "o").replace("\\`o", "o")
    s = s.replace("\\`{a}", "a").replace("\\`a", "a")
    s = s.replace("\\\"{o}", "o").replace("\\\"o", "o")
    s = s.replace("\\\"{u}", "u").replace("\\\"u", "u")
    s = s.replace("\\\"{a}", "a").replace("\\\"a", "a")
    s = s.replace("\\v{s}", "s")
    s = s.replace("\\c{c}", "c")
    s = s.replace("\\'{c}", "c")
    s = s.replace("{\'o}", "o")
    s = s.replace("{\\'a}", "a")
    s = s.replace("{\\i}", "i")
    s = s.replace("{\\'e}", "e")
    s = s.replace("{\\'o}", "o")
    s = s.replace("{\\'a}", "a")
    s = s.replace("{\\`e}", "e")
    s = s.replace("{\\`o}", "o")
    s = s.replace("{\\`a}", "a")
    s = s.replace("{\\\"o}", "o")
    s = s.replace("{\\\"u}", "u")
    s = s.replace("{\\\"a}", "a")
    s = s.replace("{\\v s}", "s")
    s = s.replace("{\\c c}", "c")
    s = s.replace("{\\'c}", "c")
    s = s.replace("\\'{c}", "c")
    s = s.replace("\\'{C}", "C")
    s = s.replace("{\\'C}", "C")
    s = s.replace("\\'{o}", "o")
    s = s.replace("\\'{O}", "O")
    s = s.replace("{\\'O}", "O")
    # Clean braces
    s = s.replace("{", "").replace("}", "")
    return s

def format_authors(author_str):
    if not author_str:
        return "Unknown"
    authors = re.split(r'\s+and\s+', author_str.strip(), flags=re.IGNORECASE)
    formatted = []
    for auth in authors:
        auth = clean_braces(auth).strip()
        if ',' in auth:
            parts = auth.split(',')
            last = parts[0].strip()
            firsts = parts[1].strip().split()
            initials = "".join([f"{f[0]}." for f in firsts if f])
            formatted.append(f"{last}, {initials}")
        else:
            parts = auth.split()
            if len(parts) > 1:
                last = parts[-1]
                firsts = parts[:-1]
                initials = "".join([f"{f[0]}." for f in firsts if f])
                formatted.append(f"{last}, {initials}")
            else:
                formatted.append(auth)
    
    if len(formatted) == 0:
        return "Unknown"
    elif len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} \\& {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + ", \\& " + formatted[-1]

def get_short_author(author_str):
    if not author_str:
        return "Unknown"
    authors = re.split(r'\s+and\s+', author_str.strip(), flags=re.IGNORECASE)
    if not authors:
        return "Unknown"
    
    first_auth = clean_latex_accents_for_short_author(authors[0]).strip()
    if ',' in first_auth:
        last = first_auth.split(',')[0].strip()
    else:
        last = first_auth.split()[-1].strip()
    
    # Strip any brackets
    last = last.replace("{", "").replace("}", "")
    
    if len(authors) == 1:
        return last
    elif len(authors) == 2:
        second_auth = clean_latex_accents_for_short_author(authors[1]).strip()
        if ',' in second_auth:
            second_last = second_auth.split(',')[0].strip()
        else:
            second_last = second_auth.split()[-1].strip()
        second_last = second_last.replace("{", "").replace("}", "")
        return f"{last} \\& {second_last}"
    else:
        return f"{last} et~al."

formatted_entries = []

for key, (etype, fields) in bib_dict.items():
    authors = format_authors(fields.get("author", fields.get("organization", "")))
    short_auth = get_short_author(fields.get("author", fields.get("organization", "")))
    
    year = fields.get("year", "n.d.")
    title = clean_braces(fields.get("title", ""))
    
    doi = fields.get("doi", "")
    doi_str = f" \\url{{https://doi.org/{doi}}}" if doi else ""
    
    if etype == "article":
        journal = clean_braces(fields.get("journal", ""))
        volume = fields.get("volume", "")
        number = fields.get("number", "")
        pages = fields.get("pages", "").replace("--", "–")
        art_num = fields.get("article-number", "")
        
        issue_str = f"({number})" if number else ""
        pages_str = f", {pages}" if pages else (f", {art_num}" if art_num else "")
        vol_str = f", \\textit{{{volume}}}" if volume else ""
        
        formatted = f"{authors} ({year}). {title}. \\textit{{{journal}}}{vol_str}{issue_str}{pages_str}.{doi_str}"
    elif etype == "techreport":
        institution = clean_braces(fields.get("institution", fields.get("publisher", "")))
        number = fields.get("number", "")
        rep_str = f" (Tech. Rep. No. {number})" if number else " (Tech. Rep.)"
        formatted = f"{authors} ({year}). \\textit{{{title}}}{rep_str}. {institution}.{doi_str}"
    elif etype == "inproceedings" or etype == "incollection":
        booktitle = clean_braces(fields.get("booktitle", ""))
        pages = fields.get("pages", "").replace("--", "–")
        pages_str = f" (pp. {pages})" if pages else ""
        publisher = clean_braces(fields.get("publisher", ""))
        formatted = f"{authors} ({year}). {title}. In \\textit{{{booktitle}}}{pages_str}. {publisher}.{doi_str}"
    else: # book or generic
        publisher = clean_braces(fields.get("publisher", ""))
        formatted = f"{authors} ({year}). \\textit{{{title}}}. {publisher}.{doi_str}"
    
    formatted = re.sub(r'\.\s*\.', '.', formatted)
    
    # Sort key clean for alphabetizing
    sort_key = clean_latex_accents_for_short_author(authors).lower()
    sort_key = re.sub(r'[^a-z0-9\s]', '', sort_key)
    sort_key = f"{sort_key} {year} {title}".lower()
    
    formatted_entries.append((sort_key, key, short_auth, year, formatted))

# Sort alphabetically
formatted_entries.sort(key=lambda x: x[0])

print(f"Total entries: {len(formatted_entries)}")

with open(r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\scratch\formatted_bib.txt", "w", encoding="utf-8") as out:
    out.write("\\begin{thebibliography}{" + str(len(formatted_entries)) + "}\n")
    out.write("\\expandafter\\ifx\\csname url\\endcsname\\relax\n")
    out.write("  \\def\\url#1{\\texttt{#1}}\\fi\n")
    out.write("\\expandafter\\ifx\\csname urlprefix\\endcsname\\relax\\def\\urlprefix{URL }\\fi\n")
    out.write("\\expandafter\\ifx\\csname href\\endcsname\\relax\n")
    out.write("  \\def\\href#1#2{#2} \\def\\path#1{#1}\\fi\n\n")
    
    for sort_key, key, short_auth, year, formatted in formatted_entries:
        label = f"{short_auth}({year})"
        label_clean = label.replace("~", " ").replace("\\&", "&")
        out.write(f"\\bibitem[{label_clean}]{{{key}}}\n")
        out.write(f"{formatted}\n\n")
    out.write("\\end{thebibliography}\n")

print("Generated formatted bibliography successfully!")
