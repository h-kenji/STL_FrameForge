def read_specs(file_path):
    """
    Reads key-value pairs from the specs file.
    """
    specs = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=")
                    specs[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit(1)
    return specs