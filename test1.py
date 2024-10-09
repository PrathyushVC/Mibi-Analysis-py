OPENSLIDE_PATH = r'C:\Users\chirr\Downloads\openslide-bin-4.0.0.3-windows-x64\bin'
import os
import sys
print(sys.executable)
print(os.listdir(OPENSLIDE_PATH))
if hasattr(os, 'add_dll_directory'):
    # Windows
    print("Running on Windows")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide