#!/bin/bash -e

version=$1

echo "Downloading the getting started package..."

url="https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-getting-started/versions/${version}/zip"
outfile="getting_started_v${version}.zip"

if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -J -L $url -o $outfile
else
    wget --content-disposition $url -O $outfile
fi

echo "Extracting the getting started package..."
unzip -u $outfile -d "./getting_started_v${version}" && rm -f $outfile

cd getting_started_v${version}

echo "Installing the getting started package..."
bash setup/quickstart_launcher.sh --install

echo "Upgrading the getting started package..."
bash setup/quickstart_launcher.sh --upgrade

echo "Done!"
