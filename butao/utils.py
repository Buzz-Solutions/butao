def get_value_from_spec(spec_file: str, key: str):
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
    key_line = [line for line in lines if key + ":" in line]
    assert len(key_line) == 1
    key_line = key_line[0]

    # get value
    value = key_line.split(" ")[-1].strip()
    return value, key_line


def adjust_spec(spec_file: str, key: str, new_value):
    """Changes the value for a given key in a spec file

    Parameters
    spec_file : str
        Path to the spec file
    key : str
        Key to look for
    new_value : str
        New value for the given key
    """
    old_value, key_line = get_value_from_spec(spec_file, key)

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
