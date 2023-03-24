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

    return cwd


def get_root_config(config="detectnet_v2"):
    """Get the root config file."""
    root_dir = get_root_dir()
    return os.path.join(root_dir, "configs", f"{config}.yaml")


class TaoEnv:
    def __init__(self, config_yaml_path, root_dir=None):
        if root_dir is None:
            root_dir = get_root_dir()

        with open(config_yaml_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.tao_model_name = self.params["TAO_MODEL_NAME"]
        self.user_workspace_dir = self.params["USER_WORKSPACE_DIR"]

        # Set local directories
        self.local_project_dir = os.path.join(
            root_dir, "notebooks", self.tao_model_name
        )
        self.local_data_dir = os.path.join(self.local_project_dir, "data")
        self.local_experiment_dir = os.path.join(
            self.local_project_dir, self.tao_model_name
        )
        self.local_specs_dir = os.path.join(self.local_project_dir, "specs")

        # set workspace directories (what's mounted on the docker container)
        self.data_download_dir = os.path.join(self.user_workspace_dir, "data")
        self.user_experiment_dir = os.path.join(
            self.user_workspace_dir, self.tao_model_name
        )
        self.specs_dir = os.path.join(
            self.user_workspace_dir, self.tao_model_name, "specs"
        )

        # Make the experiment directory
        os.makedirs(self.local_experiment_dir, exist_ok=True)

    def make_mount_file(self):
        WANDB_LOGGED_IN = False
        WANDB_API_KEY = self.params["WANDB_API_KEY"]
        if WANDB_API_KEY is not None:
            WANDB_LOGGED_IN = wandb.login()
            if WANDB_LOGGED_IN:
                print("WANDB successfully logged in.")

        # mounts is a dictionary that will be converted to a json file
        # maps the source and destination directories for the docker container
        # the source is the abs path to the local directory you want to mount
        # the destination is the abs path to the directory on the docker container

        mounts_file = os.path.expanduser("~/.tao_mounts.json")

        # Define the dictionary with the mapped drives
        drive_map = {
            "Mounts": [
                # Mapping the data directory
                {
                    "source": self.local_project_dir,
                    "destination": self.params["USER_WORKSPACE_DIR"],
                },
                # Mapping the specs directory.
                {
                    "source": self.local_specs_dir,
                    "destination": self.specs_dir,
                },
            ],
            "DockerOptions": {"user": f"{os.getuid()}:{os.getgid()}"},
        }

        if self.params["CLEARML_LOGGED_IN"]:
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

        if WANDB_LOGGED_IN:
            if "Envs" not in drive_map.keys():
                drive_map["Envs"] = []
            # Weights and biases currently requires access to the
            # /.config directory in the docker. Therefore, the docker
            # must be instantiated as root user. With the cells mentioned below
            # we will be deleting the cells that set user roles.
            if "user" in drive_map["DockerOptions"].keys():
                del drive_map["DockerOptions"]["user"]
            drive_map["Envs"].extend(
                [{"variable": "WANDB_API_KEY", "value": WANDB_API_KEY}]
            )

        print("Writing mounts file to: ", mounts_file)
        print(drive_map)

        with open(mounts_file, "w") as mfile:
            json.dump(drive_map, mfile, indent=4)
