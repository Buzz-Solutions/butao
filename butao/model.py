import yaml
import os


class TaoModel:
    def __init__(self, config_yaml_path):
        with open(config_yaml_path, "r") as f:
            env_vars = yaml.safe_load(f)

        self.ngc_registry = env_vars["NGC_REGISTRY"]
        self.model_type = env_vars["MODEL_TYPE"]
        self.num_gpus = env_vars["NUM_GPUS"]

        self.local_train_spec_init = os.path.join(
            os.environ["LOCAL_SPECS_DIR"], env_vars["TRAIN_SPEC_FILE"]
        )
        self.local_train_spec = self.local_train_spec_init

        self.local_model_dir = os.path.join(
            os.environ["LOCAL_EXPERIMENT_DIR"], f"pretrained_{self.model_type}"
        )
        os.makedirs(self.local_model_dir, exist_ok=True)

    def adjust_train_spec(self):
        """Reads the model spec file and adjusts the batch_size_per_gpu"""

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
        with open(self.local_train_spec, "w") as f:
            for line in lines:
                if "batch_size_per_gpu" in line:
                    # Replace the old value with the new value
                    line = f"{batch_size_per_gpu_line_adj}\n"
                f.write(line)

    def print_train_spec(self):
        print(f"\nModel spec file for training:, {self.local_train_spec}")
        with open(self.local_train_spec, "r") as f:
            print(f.read())
