import os

from butao.env import TaoEnv, get_root_dir, get_root_config
from butao.data import TaoData
from butao.model import TaoModel
from butao.utils import get_spec_value, set_spec_value


def test_get_root_config():
    root_config = get_root_config(config="detectnet_v2")
    assert root_config.endswith("butao/configs/detectnet_v2.yaml")


def test_env():
    env = TaoEnv("tests/sample_config.yml")
    root_dir = get_root_dir()

    assert env.params["TAO_MODEL_NAME"] == "detectnet_v2"
    assert env.params["NUM_GPUS"] == 2

    assert env.local_project_dir == f"{root_dir}/notebooks/detectnet_v2"
    assert env.local_data_dir == f"{root_dir}/notebooks/detectnet_v2/data"
    assert env.local_experiment_dir == f"{root_dir}/notebooks/detectnet_v2/detectnet_v2"
    assert env.local_specs_dir == f"{root_dir}/notebooks/detectnet_v2/specs"

    assert env.data_download_dir == "/workspace/tao-experiments/data"
    assert env.user_experiment_dir == "/workspace/tao-experiments/detectnet_v2"
    assert env.specs_dir == "/workspace/tao-experiments/detectnet_v2/specs"


def test_data():
    for config in os.listdir("configs"):
        config_path = os.path.join("configs", config)
        data = TaoData(config_path)
        assert data is not None


def test_model():
    for config in os.listdir("configs"):
        config_path = os.path.join("configs", config)
        print(config)
        model = TaoModel(config_path)
        assert model is not None


def test_get_value_from_spec():
    value, key_line = get_spec_value("tests/sample_spec.txt", "image_width")
    assert value == "1248"
    assert key_line == "  image_width: 1248\n"


def test_adjust_spec():
    with open("tests/sample_spec.txt", "r") as f:
        with open("tests/sample_spec_adjusted.txt", "w") as f2:
            f2.write(f.read())

    set_spec_value("tests/sample_spec_adjusted.txt", "image_width", 1249)
    value, key_line = get_spec_value("tests/sample_spec_adjusted.txt", "image_width")
    assert value == "1249"
    assert key_line == "  image_width: 1249\n"
