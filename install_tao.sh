#!/bin/bash

version=$1

echo "Downloading the getting started package..."
wget --content-disposition "https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-getting-started/versions/${version}/zip" -O "getting_started_v${version}.zip"

echo "Extracting the getting started package..."
unzip -u "getting_started_v${version}.zip" -d "./getting_started_v${version}" && rm -rf "getting_started_v${version}.zip"

echo "Installing the getting started package..."
bash getting_started_v${version}/setup/quickstart_launcher.sh --install

echo "Upgrading the getting started package..."
bash getting_started_v${version}/setup/quickstart_launcher.sh --upgrade

echo "Done!"
