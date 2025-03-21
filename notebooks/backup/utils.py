import os
from datetime import datetime

def generate_unique_filename(base_name):
    """
    Generate a unique filename by adding a timestamp to the base name.
    
    Args:
        base_name (str): The original filename (e.g., 'chart.png')
        
    Returns:
        str: A unique filename with timestamp (e.g., 'chart_20230415_123045.png')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(base_name)
    return f"{name}_{timestamp}{ext}" 