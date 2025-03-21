import re
import json
import os
from datetime import datetime

def add_unique_filename_function(notebook_path):
    """
    Add the generate_unique_filename function to the notebook if it doesn't exist.
    
    Args:
        notebook_path (str): Path to the notebook file
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check if the function already exists
    function_exists = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def generate_unique_filename' in source:
                function_exists = True
                break
    
    # If the function doesn't exist, add it after the imports
    if not function_exists:
        # Find a suitable cell to add the function (after imports)
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and ('import' in ''.join(cell['source'])):
                # Create a new cell with the function
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Function to generate unique filenames with timestamps\n",
                        "def generate_unique_filename(base_name):\n",
                        "    \"\"\"\n",
                        "    Generate a unique filename by adding a timestamp to the base name.\n",
                        "    \n",
                        "    Args:\n",
                        "        base_name (str): The original filename (e.g., 'chart.png')\n",
                        "        \n",
                        "    Returns:\n",
                        "        str: A unique filename with timestamp (e.g., 'chart_20230415_123045.png')\n",
                        "    \"\"\"\n",
                        "    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                        "    name, ext = os.path.splitext(base_name)\n",
                        "    return f\"{name}_{timestamp}{ext}\"\n"
                    ],
                    "outputs": []
                }
                notebook['cells'].insert(i+1, new_cell)
                break
    
    # Update all file_name assignments
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            updated_source = []
            for line in cell['source']:
                # Match lines like: file_name = 'something.png'
                match = re.match(r"(\s*file_name\s*=\s*)['\"]([^'\"]+)['\"](.*)$", line)
                if match:
                    indent, filename, rest = match.groups()
                    updated_line = f"{indent}generate_unique_filename('{filename}'){rest}\n"
                    updated_source.append(updated_line)
                else:
                    updated_source.append(line)
            cell['source'] = updated_source
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated {notebook_path} with unique filename generation.")

if __name__ == "__main__":
    notebook_path = "notebooks/main.ipynb"
    add_unique_filename_function(notebook_path) 