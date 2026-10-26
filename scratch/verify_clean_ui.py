import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'D:\3dgs-incremental-research\ui\app.py', 'r', encoding='utf-8') as f:
    code = f.read()

tabs_match = re.search(r'st\.tabs\(\[\s*([\s\S]*?)\]\)', code)
if tabs_match:
    print('Tabs configured:')
    for line in tabs_match.group(1).strip().splitlines():
        line = line.strip().strip(',').strip('"').strip("'")
        if line:
            print(f'  - {line}')

print('\nButtons configured:')
buttons = re.findall(r'\.button\(\s*"([^"]+)"', code)
for b in buttons:
    print(f'  - {b}')

print('\nExpanders configured:')
expanders = re.findall(r'st\.expander\(\s*"([^"]+)"', code)
for e in expanders:
    print(f'  - {e}')
