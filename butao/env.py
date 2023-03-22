import json
import os
import yaml
import wandb


def get_root_dir():
    """Get the root directory of the project."""
    # Get the current working directory
    cwd = os.getcwd()

    # Traverse up the directory tree until you find the ".git" folder
    while not os.path.exists(os.path.join(cwd, ".git")):
        cwd = os.path.dirname(cwd)

    root_dir = cwd
    return root_dir


def get_root_config():
    """Get the root config file."""
    root_dir = get_root_dir()
    config_path = os.path.join(root_dir, "configs", "env_config.yaml")
    return config_path


def set_environment(config_yaml_path, root_dir=None):
    """Set environment variables for the current session."""
    with open(config_yaml_path, "r") as f:
        env_vars = yaml.safe_load(f)

    os.environ["TAO_MODEL_NAME"] = env_vars["TAO_MODEL_NAME"]
    os.environ["KEY"] = env_vars["KEY"]

    assert type(env_vars["NUM_GPUS"]) == int
    os.environ["NUM_GPUS"] = str(env_vars["NUM_GPUS"])

    # Set local directories
    os.environ["LOCAL_PROJECT_DIR"] = os.path.join(
        root_dir, "notebooks", env_vars["TAO_MODEL_NAME"]
    )
    os.environ["LOCAL_DATA_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "data")
    os.environ["LOCAL_EXPERIMENT_DIR"] = os.path.join(
        os.environ["LOCAL_PROJECT_DIR"], env_vars["TAO_MODEL_NAME"]
    )
    os.environ["LOCAL_SPECS_DIR"] = os.path.join(
        os.environ["LOCAL_PROJECT_DIR"], "specs"
    )

    # set workspace directories (what's mounted on the docker container)
    os.environ["DATA_DOWNLOAD_DIR"] = os.path.join(
        env_vars["USER_WORKSPACE_DIR"], "data"
    )
    os.environ["USER_EXPERIMENT_DIR"] = os.path.join(
        env_vars["USER_WORKSPACE_DIR"], env_vars["TAO_MODEL_NAME"]
    )
    os.environ["SPECS_DIR"] = os.path.join(
        env_vars["USER_WORKSPACE_DIR"], env_vars["TAO_MODEL_NAME"], "specs"
    )

    # Make the experiment directory
    os.makedirs(os.environ["LOCAL_EXPERIMENT_DIR"], exist_ok=True)

    # WANDB_API_KEY must be set as an environment variable for wandb to work
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
    if WANDB_API_KEY is not None:
        WANDB_LOGGED_IN = wandb.login()
        if WANDB_LOGGED_IN:
            print("WANDB successfully logged in.")

    # mounts is a dictionary that will be converted to a json file
    # maps the source and destination directories for the docker container
    # the source should be the abs path to the local directory you want to mount
    # the destination should be the abs path to the directory on the docker container

    mounts_file = os.path.expanduser("~/.tao_mounts.json")

    # Define the dictionary with the mapped drives
    drive_map = {
        "Mounts": [
            # Mapping the data directory
            {
                "source": os.environ["LOCAL_PROJECT_DIR"],
                "destination": env_vars["USER_WORKSPACE_DIR"],
            },
            # Mapping the specs directory.
            {
                "source": os.environ["LOCAL_SPECS_DIR"],
                "destination": os.environ["SPECS_DIR"],
            },
        ],
        "DockerOptions": {"user": f"{os.getuid()}:{os.getgid()}"},
    }

    if env_vars["CLEARML_LOGGED_IN"]:
        if "Envs" not in drive_map.keys():
            drive_map["Envs"] = []
        drive_map["Envs"].extend(
            [
                {
                    "variable": "CLEARML_WEB_HOST",
                    "value": os.getenv("CLEARML_WEB_HOST"),
                },
                {
                    "variable": "CLEARML_API_HOST",
                    "value": os.getenv("CLEARML_API_HOST"),
                },
                {
                    "variable": "CLEARML_FILES_HOST",
                    "value": os.getenv("CLEARML_FILES_HOST"),
                },
                {
                    "variable": "CLEARML_API_ACCESS_KEY",
                    "value": os.getenv("CLEARML_API_ACCESS_KEY"),
                },
                {
                    "variable": "CLEARML_API_SECRET_KEY",
                    "value": os.getenv("CLEARML_API_SECRET_KEY"),
                },
            ]
        )

    if env_vars["WANDB_LOGGED_IN"]:
        if "Envs" not in drive_map.keys():
            drive_map["Envs"] = []
        # Weights and biases currently requires access to the
        # /.config directory in the docker. Therefore, the docker
        # must be instantiated as root user. With the cells mentioned below
        # we will be deleting the cells that set user roles.
        if "user" in drive_map["DockerOptions"].keys():
            del drive_map["DockerOptions"]["user"]
        drive_map["Envs"].extend(
            [{"variable": "WANDB_API_KEY", "value": os.getenv("WANDB_API_KEY")}]
        )

    # Writing the mounts file.
    with open(mounts_file, "w") as mfile:
        json.dump(drive_map, mfile, indent=4)


if __name__ == "__main__":
    root_dir = get_root_dir()
    config_path = get_root_config()
    set_environment(config_path, root_dir=root_dir)
