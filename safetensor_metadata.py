import argparse
import json
import struct
import os

def extract_safetensor_metadata(file_path):
    """
    Reads metadata from a .safetensors file header or a standalone .json file.
    """
    _, extension = os.path.splitext(file_path.lower())

    if extension == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif extension == ".safetensors":
        with open(file_path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                raise IOError("Invalid safetensors file. Could not read header length.")
            
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            
            json_header_bytes = f.read(header_len)
            if len(json_header_bytes) != header_len:
                raise IOError("Invalid safetensors file. Header is truncated.")
                
            json_header_str = json_header_bytes.decode('utf-8')
            return json.loads(json_header_str)
    else:
        raise ValueError(f"Unsupported file type: {extension}. Must be .safetensors or .json")

def save_metadata_to_json(metadata, output_path):
    """
    Saves a dictionary to a JSON file with indentation.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

def main():
    """
    Main function to parse arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Extract metadata from a .safetensors or .json file and save to a .json file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .safetensors or .json file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata.json",
        help="Path to the output .json file."
    )
    args = parser.parse_args()

    try:
        metadata = extract_safetensor_metadata(args.input_file)
        save_metadata_to_json(metadata, args.output)
        print(f"Successfully extracted metadata to {args.output}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()