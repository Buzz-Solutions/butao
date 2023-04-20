import os
import matplotlib.pyplot as plt
from PIL import Image

from butao.env import TaoEnv
from butao.utils import get_spec_value, set_spec_value, get_model_fn_from_dir

valid_image_ext = [".jpg", ".png", ".jpeg", ".ppm"]


class TaoModel(TaoEnv):
    def __init__(self, config_yaml_path):
        TaoEnv.__init__(self, config_yaml_path)

        self.ngc_registry = self.params["NGC_REGISTRY"]
        self.model_version = self.params["MODEL_VERSION"]
        self.key = self.params["KEY"]
        self.num_gpus = self.params["NUM_GPUS"]
        assert type(self.num_gpus) == int

        # training variables
        self.train_spec_init = os.path.join(
            self.specs_dir, self.params["TRAIN_SPEC_FILE"]
        )
        self.local_train_spec_init = os.path.join(
            self.local_specs_dir, self.params["TRAIN_SPEC_FILE"]
        )

        self.train_spec = self.train_spec_init
        self.local_train_spec = self.local_train_spec_init

        self.model_dir = self.ngc_registry.split("/")[-1] + "_v" + self.model_version
        self.local_model_dir = os.path.join(self.local_experiment_dir, self.model_dir)
        self.user_model_dir = os.path.join(self.user_experiment_dir, self.model_dir)

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
        print(f"\nModel spec file for training: {self.local_train_spec}")

        # inference variables

        self.inference_spec = os.path.join(
            self.specs_dir, self.params["INFERENCE_SPEC_FILE"]
        )
        self.local_inference_spec = os.path.join(
            self.local_specs_dir, self.params["INFERENCE_SPEC_FILE"]
        )

        self.sample_data_dir = os.path.join(self.data_download_dir, "test_samples")
        self.local_sample_data_dir = os.path.join(self.local_data_dir, "test_samples")

        self.infer_dir = os.path.join(self.user_experiment_dir, "tlt_infer_testing")

        self.local_annotated_dir = os.path.join(
            self.local_experiment_dir, "tlt_infer_testing", "images_annotated"
        )
        self.local_predictions_dir = os.path.join(
            self.local_experiment_dir, "tlt_infer_testing", "labels"
        )
        os.makedirs(self.local_sample_data_dir, exist_ok=True)
        print(f"\nModel spec file for inference: {self.local_inference_spec}")

    def _get_model_fp_quoted(self):
        self.model_fn = get_model_fn_from_dir(self.local_model_dir)
        user_model_fp = os.path.join(self.user_model_dir, self.model_fn)
        self.user_model_fp_quoted = '"' + user_model_fp + '"'
        self.user_trt_fp_quoted = '"' + user_model_fp + '.trt"'

    def print_train_hyperparams(
        self, hparams=("min_learning_rate", "max_learning_rate", "batch_size_per_gpu")
    ):
        """Reads the model spec file and prints the listed hyperparams

        Args:
            hparams (list): list of hyperparams to return
        """
        print(f"spec file: {self.local_train_spec_init}")
        for hp in hparams:
            value, _ = get_spec_value(self.local_train_spec_init, hp)
            print(f"{hp}: {value}")

    def update_train_spec(self, hparams: dict = None):
        """Updates model spec file using current model file and the input hyperparams

        Args:
            hparams (dict): dictionary of hyperparams to adjust.
             Example: {
                        "min_learning_rate": "5e-05",
                        "max_learning_rate": "5e-03",
                        "batch_size_per_gpu": 8
                        }
        """
        self._get_model_fp_quoted()
        set_spec_value(
            self.local_train_spec_init,
            "pretrained_model_file",
            self.user_model_fp_quoted,
        )

        if hparams is None:
            return

        assert type(hparams) == dict

        self.local_train_spec = self.local_train_spec_init.replace(
            ".txt", "_adjusted.txt"
        )
        with open(self.local_train_spec_init, "r") as f:
            with open(self.local_train_spec, "w") as f2:
                f2.write(f.read())
        print("created adjusted spec file: ", self.local_train_spec)

        self.train_spec = self.train_spec_init.replace(".txt", "_adjusted.txt")

        assert type(hparams) == dict
        for hp, value in hparams.items():
            if value is not None:
                set_spec_value(self.local_train_spec, hp, value)

    def update_inference_spec(self):
        self._get_model_fp_quoted()
        if self.model_fn.endswith(".etlt"):
            set_spec_value(
                self.local_inference_spec, "etlt_model", self.user_model_fp_quoted
            )
            set_spec_value(
                self.local_inference_spec, "trt_engine", self.user_trt_fp_quoted
            )
        elif self.model_fn.endswith(".tlt"):
            set_spec_value(
                self.local_inference_spec, "tlt_model", self.user_model_fp_quoted
            )
        else:
            raise ValueError("Model file must be .etlt or .tlt")

    def plot_image_grid(self, rows, cols, num_images=10):
        filepaths = [
            os.path.join(self.local_annotated_dir, image)
            for image in os.listdir(self.local_annotated_dir)
            if os.path.splitext(image)[1].lower() in valid_image_ext
        ]

        num_images = len(filepaths) if not num_images < len(filepaths) else num_images
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        n = 0
        for i in range(rows):
            for j in range(cols):
                image = Image.open(filepaths[n])
                ax[i][j].imshow(image)
                image.close()
                ax[i][j].axis("off")
                n += 1
                if n >= num_images:
                    break
        plt.tight_layout()
        plt.show()
