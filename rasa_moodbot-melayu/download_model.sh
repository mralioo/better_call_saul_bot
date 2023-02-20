#!/bin/bash

mkdir .cache
# from https://huggingface.co/StevenLimcorn/MelayuBERT/tree/main
wget -P .cache https://huggingface.co/StevenLimcorn/MelayuBERT/resolve/main/config.json \
    https://huggingface.co/StevenLimcorn/MelayuBERT/resolve/main/special_tokens_map.json \
    https://huggingface.co/StevenLimcorn/MelayuBERT/resolve/main/tf_model.h5 \
    https://huggingface.co/StevenLimcorn/MelayuBERT/resolve/main/tokenizer_config.json \
    https://huggingface.co/StevenLimcorn/MelayuBERT/resolve/main/vocab.txt