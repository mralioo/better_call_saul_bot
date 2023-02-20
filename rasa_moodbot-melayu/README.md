# Introduction

This repo shows how to add a new [hugging face model](https://huggingface.co/models) as a featurizer by subclassing the [LanguageModelFeaturizer](https://rasa.com/docs/rasa/components#languagemodelfeaturizer).

It the model used to demonstrate this approach is the [MelayuBERT](https://huggingface.co/StevenLimcorn/MelayuBERT).

Details about this approach is explained in this [blog post](https://pub.towardsai.net/how-to-use-rasa-to-build-a-bot-that-understands-bahasa-melayu-ab97bcb6f546).

# Prerequisites
1. Docker
2. VS Code

# Usage

1. Run `bash download_model.sh`
2. Run `rasa train`
3. Run `make run-bot`
4. Preview the [bot_ui.html](bot_ui.html) file
5. Chat with the bot