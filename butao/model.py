import os

from butao.env import TaoEnv


class TaoModel(TaoEnv):
    def __init__(self, config_yaml_path):
        TaoEnv.__init__(self, config_yaml_path)

        self.ngc_registry = self.params["NGC_REGISTRY"]
        self.model_type = self.params["MODEL_TYPE"]
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

        self.local_model_dir = os.path.join(
            self.local_experiment_dir, f"pretrained_{self.model_type}"
        )

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

    def adjust_train_spec(self):
        """Reads the model spec file and adjusts the batch_size_per_gpu
        batch_size_per_gpu = batch_size_per_gpu // num_gpus
        """

        with open(self.local_train_spec_init, "r") as f:
            lines = f.readlines()

        batch_size_per_gpu_line = [
            line for line in lines if "batch_size_per_gpu" in line
        ]
        assert len(batch_size_per_gpu_line) == 1
        batch_size_per_gpu_line = batch_size_per_gpu_line[0]
        batch_size_per_gpu = int(batch_size_per_gpu_line.split(" ")[-1].strip())

        batch_size_per_gpu_line_adj = batch_size_per_gpu_line.replace(
            str(batch_size_per_gpu), str(batch_size_per_gpu // self.num_gpus)
        )

        self.local_train_spec = self.local_train_spec_init.replace(
            ".txt", "_adjusted.txt"
        )
        self.train_spec = self.train_spec_init.replace(".txt", "_adjusted.txt")
        with open(self.local_train_spec, "w") as f:
            for line in lines:
                if "batch_size_per_gpu" in line:
                    # Replace the old line with the new line
                    line = batch_size_per_gpu_line_adj
                f.write(line)

    def print_train_spec(self):
        print(f"\nModel spec file for training:, {self.local_train_spec}")
        with open(self.local_train_spec, "r") as f:
            print(f.read())
