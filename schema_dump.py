import sys
try:
    from pydantic import BaseModel
except ImportError:
    print("pydantic not found")
    sys.exit(1)

try:
    # Look for TaskSpec in openenv
    found = False
    import openenv
    import os
    
    # Just list all files in the package and look for definitions of TaskSpec
    pkg_dir = os.path.dirname(openenv.__file__)
    for root, dirs, files in os.walk(pkg_dir):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as file:
                    if 'class Task' in file.read() or 'TaskSpec' in file.read():
                        print(f"Found something in {path}")
                        
except Exception as e:
    print(e)
