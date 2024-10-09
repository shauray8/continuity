import struct
import zlib
import json
from pathlib import Path

def create_cnt_file(data: dict, file_path: Path) -> None:
    """Create a .cnt file from user-provided data."""
    with open(file_path, 'wb') as f:
        for key, value in data.items():
            # Prepare the section data
            section_data = f"{key}:{value}".encode('utf-8')

            # Calculate the size of the section data
            section_size = len(section_data)

            # Calculate CRC for the section data
            crc = zlib.crc32(section_data)

            # Create the section
            section = struct.pack("I", section_size)  # Size
            section += struct.pack("B", 1)  # CRC Check flag (1 = check)
            section += section_data  # Actual data
            section += struct.pack("I", crc)  # Append the CRC

            # Write the section to the file
            f.write(section)

    print(f".cnt file created at: {file_path}")

if __name__ == "__main__":
    # Example usage
    user_data = {
        "model": "FLUX",
        "lora_adapters": "lora_adapter_1:0.5",
        "controlnet": "depth_map:1.0",
        "use_memory_efficient_attention": "true"
    }
    
    create_cnt_file(user_data, Path("output_file.cnt"))

