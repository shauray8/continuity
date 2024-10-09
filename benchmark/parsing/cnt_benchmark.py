import yaml
import time
import os
import zlib
import struct
import concurrent.futures
from pathlib import Path

# -------------- Mock `.cnt` Parsing Functions -------------- #

def parse_cnt_section(section: bytes) -> dict:
    """Mock parser for a section in `.cnt` format."""
    # Assume section contains a simple key-value pair
    header_size = struct.unpack("I", section[:4])[0]  # read header size
    crc_from_file = struct.unpack("I", section[-4:])[0]  # CRC checksum
    data = section[4:-4]
    
    # Verify CRC
    if zlib.crc32(data) != crc_from_file:
        raise ValueError("CRC Check Failed: Corrupted Section")

    # Decode the data (assuming it's a key-value pair in binary)
    key, value = data.split(b':', 1)
    return {key.decode(): value.decode()}

def parse_cnt_file(file_path: Path, parallel: bool = False) -> dict:
    """Parse `.cnt` file section by section."""
    with open(file_path, "rb") as f:
        sections = []
        while True:
            size_data = f.read(4)  # First 4 bytes contain the section size
            if not size_data:
                break

            section_size = struct.unpack("I", size_data)[0]
            section_data = f.read(section_size + 4)  # Read section (data + CRC)
            if len(section_data) != section_size + 4:
                raise ValueError("Section size mismatch or file corruption detected")

            sections.append(size_data + section_data)  # Append the whole section for processing

    results = {}
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            parsed_sections = list(executor.map(parse_cnt_section, sections))
        for result in parsed_sections:
            results.update(result)
    else:
        for section in sections:
            result = parse_cnt_section(section)
            results.update(result)

    return results

def parse_cnt_section(section: bytes) -> dict:
    """Mock parser for a section in `.cnt` format."""
    # Ensure the section has at least 8 bytes (4 bytes for size and 4 for CRC)
    if len(section) < 8:
        raise ValueError("Section too small to contain valid data and CRC")

    # Section contains size (4 bytes), data, and CRC (4 bytes)
    section_size = struct.unpack("I", section[:4])[0]  # Get section size from the header

    # Check if section has enough data (section_size + 4 bytes for CRC)
    expected_total_size = 4 + section_size + 4  # size field (4 bytes) + data + CRC (4 bytes)
    if len(section) < expected_total_size:
        raise ValueError("Section too small for declared size and CRC")

    data = section[4:4+section_size]  # Read the actual data
    crc_from_file = struct.unpack("I", section[4+section_size:4+section_size+4])[0]  # Extract CRC at the end

    # Verify CRC
    if zlib.crc32(data) != crc_from_file:
        raise ValueError("CRC Check Failed: Corrupted Section")

    # Decode the data (assuming it's a key-value pair in binary format)
    key, value = data.split(b':', 1)  # Splitting by ':' to get key-value
    return {key.decode(): value.decode()}

def create_mock_cnt_file(file_path: Path, num_sections: int = 100):
    """Create a mock `.cnt` file with multiple sections and CRC checks."""
    with open(file_path, "wb") as f:
        for i in range(num_sections):
            key = f"param{i}".encode()
            value = f"value{i}".encode()
            data = key + b":" + value  # Example format: key:value

            # Calculate section size (data only, excluding size and CRC fields)
            section_size = len(data)

            # Compute CRC for the section data
            crc_value = zlib.crc32(data)

            # Write section size (4 bytes), followed by data and CRC (4 bytes)
            f.write(struct.pack("I", section_size))  # header: section size (4 bytes)
            f.write(data)                            # actual data
            f.write(struct.pack("I", crc_value))     # CRC checksum (4 bytes)


# -------------- YAML Parsing Functions -------------- #

def parse_yaml_file(file_path: Path) -> dict:
    """Parse a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def create_mock_yaml_file(file_path: Path, num_sections: int = 100):
    """Create a mock YAML file with multiple sections."""
    data = {f"param{i}": f"value{i}" for i in range(num_sections)}
    with open(file_path, "w") as f:
        yaml.dump(data, f)

# -------------- Benchmarking -------------- #

def benchmark(file_yaml: Path, file_cnt: Path, parallel: bool = False):
    # Benchmark YAML
    start = time.time()
    parse_yaml_file(file_yaml)
    yaml_time = time.time() - start
    
    # Benchmark `.cnt` (with or without parallelism)
    start = time.time()
    parse_cnt_file(file_cnt, parallel=parallel)
    cnt_time = time.time() - start
    
    return yaml_time, cnt_time

# -------------- Main Script -------------- #

if __name__ == "__main__":
    # File paths
    yaml_file = Path("mock_config.yaml")
    cnt_file = Path("mock_config.cnt")
    
    # Create mock files
    num_sections = 1000
    create_mock_yaml_file(yaml_file, num_sections)
    create_mock_cnt_file(cnt_file, num_sections)
    
    # Sequential Benchmark
    yaml_time, cnt_time = benchmark(yaml_file, cnt_file, parallel=False)
    print(f"Sequential Parsing: YAML = {yaml_time:.4f}s, .cnt = {cnt_time:.4f}s")
    
    # Parallel Benchmark
    yaml_time, cnt_time = benchmark(yaml_file, cnt_file, parallel=True)
    print(f"Parallel Parsing: YAML = {yaml_time:.4f}s, .cnt = {cnt_time:.4f}s")

