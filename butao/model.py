import os

from butao.env import TaoEnv
from butao.utils import get_value_from_spec, adjust_spec


class TaoModel(TaoEnv):
    def __init__(self, config_yaml_path):
        TaoEnv.__init__(self, config_yaml_path)

        self.ngc_registry = self.params["NGC_REGISTRY"]
        self.model_version = self.params["MODEL_VERSION"]
        self.key = self.params["KEY"]
        self.num_gpus = self.params["NUM_GPUS"]
        assert type(self.num_gpus) == int

        self.train_spec_init = os.path.join(
            self.specs_dir, self.params["TRAIN_SPEC_FILE"]
        )
        self.local_train_spec_init = os.path.join(
            self.local_specs_dir, self.params["TRAIN_SPEC_FILE"]
        )

        self.train_spec = self.train_spec_init
        self.local_train_spec = self.local_train_spec_init

        model_dir = self.ngc_registry.split("/")[-1] + "_v" + self.model_version
        self.local_model_dir = os.path.join(self.local_experiment_dir, model_dir)
        self.user_model_dir = os.path.join(self.user_experiment_dir, model_dir)

        self.user_experiment_dir_unpruned = os.path.join(
            self.user_experiment_dir, "experiment_dir_unpruned"
        )
        self.local_experiment_dir_unpruned = os.path.join(
            self.local_experiment_dir, "experiment_dir_unpruned"
        )
        self.user_experiment_dir_pruned = os.path.join(
            self.user_experiment_dir, "experiment_dir_pruned"
        )
        self.local_experiment_dir_pruned = os.path.join(
            self.local_experiment_dir, "experiment_dir_pruned"
        )

        os.makedirs(self.local_model_dir, exist_ok=True)
        print(f"\nModel spec file for training: {self.local_train_spec}\n")

    def show_hyperparams(
        self, hparams=("min_learning_rate", "max_learning_rate", "batch_size_per_gpu")
    ):
        """Reads the model spec file and returns the listed hyperparams

        Args:
            hparams (list): list of hyperparams to return
        """
        print(f"spec file: {self.local_train_spec_init}")
        for hp in hparams:
            value, _ = get_value_from_spec(self.local_train_spec_init, hp)
            print(f"{hp}: {value}")

    def adjust_train_spec(self, hparams={"max_learning_rate": None}):
        """Adjusts model spec file using the given values of the non-None hyperparams

        Args:
            hparams (dict): dictionary of hyperparams to adjust
        """
        self.local_train_spec = self.local_train_spec_init.replace(
            ".txt", "_adjusted.txt"
        )
        with open(self.local_train_spec_init, "r") as f:
            with open(self.local_train_spec, "w") as f2:
                f2.write(f.read())
        print("created adjusted spec file: ", self.local_train_spec)

        self.train_spec = self.train_spec_init.replace(".txt", "_adjusted.txt")

        for hp, value in hparams.items():
            if value is not None:
                adjust_spec(self.local_train_spec, hp, value)
