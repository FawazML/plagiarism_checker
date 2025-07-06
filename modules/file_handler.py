# file_handler.py
import os
import glob
from typing import List, Dict, Tuple

class FileHandler:
    def __init__(self, directory: str = None, file_extension: str = '.txt'):
        self.directory = directory or os.getcwd()
        self.file_extension = file_extension
    
    def get_file_list(self) -> List[str]:
        """Get a list of all files with the specified extension in the directory."""
        file_pattern = os.path.join(self.directory, f'*{self.file_extension}')
        return glob.glob(file_pattern)
    
    def load_files(self) -> Dict[str, str]:
        """Load the content of all files with the specified extension."""
        file_list = self.get_file_list()
        file_contents = {}
        
        for file_path in file_list:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_contents[filename] = file.read()
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
        
        return file_contents