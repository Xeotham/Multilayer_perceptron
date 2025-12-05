#!/bin/sh

if [ ! -d ./.venv ]; then \
  echo "Creating Venv..."; \
  python3 -m venv .venv; \
  .venv/bin/pip install -r ./requirement.txt ; \
  echo "Venv created." ; \
fi

if [ ! -d ./datasets -o ! -f ./datasets/dataset_test.csv -o ! -f ./dataset_train.csv ]; then \
  echo "Importing Datasets..."; \
  mkdir ./data ;\
  wget https://cdn.intra.42.fr/document/document/42613/data.csv; \
  mv data.csv ./data/data.csv ; \
  echo "Datasets imported !"; \
fi

. ./.venv/bin/activate
echo "Venv sourced."