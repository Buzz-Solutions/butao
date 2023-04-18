# butao
BUzz TAO tools

NVIDIA TAO (previously named TLT) is the SDK that nvidia provides for building AI applications using transfer learning.

It allows you to access pre-trained models stored on NGC (NVIDIA GPU Cloud) and retrain them.

The SDK runs on pre-built, containerized applications that manage most of the low-level dependencies. These can be executed on command-line or through a notebook.

The SDK is built on top of the NVIDIA Deep Learning Frameworks (NVIDIA DL Frameworks) and NVIDIA CUDA-X AI libraries.

For deployment, models integrate with NVIDIA DeepSream SDK and NVIDIA Triton Inference Server.

## Install butao

From the root level of this repo:

```
bash setup.sh
```

## Set up and install TAO

The following is a condensed version of [NVIDIA's quickstart guide](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html) and other small bits of info I found elsewhere to help fill in some gaps. If you run into issues with this setup, try restarting using the guide in full.

### Registering with NGC
#### Get an NGC account and API key:
- [Sign in to NGC](https://ngc.nvidia.com/) with your email address or click create an account and then sign in.
- Click on your username in the top right corner and select Setup.
- Click on Generate API Key.
- Copy the API key and save it in a secure location.
#### Log in to the NGC docker registry:
Run `docker login nvcr.io` and enter the following credentials. For the username, enter '$oauthtoken' exactly as shown. It is a special authentication token for all users.
```
Username: $oauthtoken
Password: <YOUR NGC API KEY>
```

### Install TAO
Check for the latest version of the getting started package here:
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-getting-started

Then set the version number like so:
```
version=X.X.X
```

Then run:
```
bash install_tao.sh $version
```

Check the output to be sure that it installed all packages successfully.

You should also see a new directory structure like this:
```
getting_started_vX.X.X/
ngc-cli  notebooks  setup
```

The notebooks folder contains a bunch of examples from NVIDIA. Check here to see which notebooks can be used for which types of models:
https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html#run-sample-jupyter-notebooks

If you just want to view the NVIDIA notebooks, you can also do so more easily [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/cv_samples/files)

The `tao_modeling` notebook in this repo was constructed with those as templates.

## Get started

1. Choose a model architecture based on one of the following lists [here](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html#run-sample-jupyter-notebooks) or [here](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models)
2. Find the model info in the ngc registry
   - Navigate to https://catalog.ngc.nvidia.com/models
   - Search for and select your model card
   - Click the 'Download' drop down in the upper right and select `wget`
   - It should copy to your clipboard a link formatted like https://api.ngc.nvidia.com/v2/models/nvidia/tao/<model_name>/versions
   - Use this <model_name> for the `NGC_REGISTRY` in your corresponding yaml configuration in `configs/`
   - Find one of the model versions listed in the model card and use as `MODEL_VERSION`
   - Find the 'Model load key' and use as `KEY`
3. Create or update the corresponding yaml configuration in `configs/`
4. You can now run all steps for model training in `notebooks/tao_modeling.ipynb`

