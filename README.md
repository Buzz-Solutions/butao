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

## Get started with TAO

The following is a condensed version of [NVIDIA's quickstart guide](https://docs.nvidia.com/metropolis/TAO/tao-user-guide/index.html#quickstart) and other small bits of info I found elsewhere to help fill in some gaps. If you run into issues with this setup, try restarting using the guide in full.

### Registering with NGC
#### Get an NGC account and API key:
- [Sign in to NGC](https://ngc.nvidia.com/) with your email address or click create an account and then sign in.
- Click on your username in the top right corner and select Setup.
- Click on Generate API Key.
- Copy the API key and save it in a secure location.
#### Log in to the NGC docker registry:
Run `docker login nvcr.io` and enter the following credentials:
- Username: "$oauthtoken"
- Password: "YOUR_NGC_API_KEY"

where YOUR_NGC_API_KEY corresponds to your key.

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

You should also now have a directory structure like this:
```
getting_started_vX.X.X/
ngc-cli  notebooks  setup
```

The notebooks folder contains a bunch of examples that you can run to get started with TAO. Check here to see which notebooks can be used for which types of models:
https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html#run-sample-jupyter-notebooks
