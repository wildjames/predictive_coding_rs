#!/usr/bin/env bash
set -e

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

mkdir -p data/mnist
cd data/mnist

for file in "${FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Downloading $file..."
    curl -O "$BASE_URL/$file"
  else
    echo "$file already exists, skipping."
  fi

  # Unzip the file if it hasn't been unzipped yet
  if [ ! -f "${file%.gz}" ];
  then
    echo "Unzipping $file..."
    gunzip -k "$file"
  else
    echo "${file%.gz} already exists, skipping unzip."
  fi
done

echo "Done."
