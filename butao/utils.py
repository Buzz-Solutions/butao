from pathlib import Path


def get_spec_value(spec_file: str, key: str):
    """Gets the value for a given key in a spec file

    Parameters
    spec_file : str
        Path to the spec file
    key : str
        Key to look for

    Returns
    value : str
        Value for the given key
    key_line : str
        Line in the spec file that contains the key
    """

    with open(spec_file, "r") as f:
        lines = f.readlines()

    # get line with key
    key_line = [line for line in lines if key + ":" == line.split()[0]]
    assert len(key_line) == 1
    key_line = key_line[0]

    # get value
    value = key_line.split()[-1]
    return value, key_line


def set_spec_value(spec_file: str, key: str, new_value):
    """Changes the value for a given key in a spec file

    Parameters
    spec_file : str
        Path to the spec file
    key : str
        Key to look for
    new_value : str
        New value for the given key
    """
    old_value, key_line = get_spec_value(spec_file, key)

    # replace value
    key_line_adj = key_line.replace(": " + str(old_value), ": " + str(new_value))

    # write new spec file
    with open(spec_file, "r") as f:
        lines = f.readlines()
    with open(spec_file, "w") as f:
        for line in lines:
            if key + ":" in line:
                # Replace the old line with the new line
                line = key_line_adj
            f.write(line)

    print(f"Changed value for {key} from {old_value} to {new_value}")


def get_model_fn_from_dir(folder_path):
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix in [".tlt", ".etlt", ".hdf5"]:
            return file_path.name
