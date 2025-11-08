
# AI Inference with Hugging Face Pipelines

This repository contains a comprehensive Jupyter Notebook, `Sentiment_Analysis_and_HF_Pipelines.ipynb`, designed as a hands-on to leveraging the high-level `pipeline` API from the Hugging Face `transformers` and `diffusers` libraries. This notebook is a gateway to performing a wide array of powerful AI tasks—from Natural Language Processing (NLP) to image and audio generation—with just a few lines of Python code.

## 🚀 Project Overview

The Hugging Face `pipeline` API is an abstraction layer that simplifies the process of using pre-trained models for **inference**. It handles the complex plumbing of tokenization, model loading, and post-processing, allowing you to focus on the application rather than the architecture.

This notebook serves as a practical, code-first tutorial that walks you through:
- The core concepts of AI model **training vs. inference**.
- The simple, two-step workflow of the `pipeline` API.
- Hands-on examples for a diverse set of cutting-edge AI tasks.
- Important tips and best practices for working within a Google Colab environment.

## ✨ Featured Pipelines & Tasks

This notebook provides working examples for the following AI tasks, showcasing the versatility of the `pipeline` API:

### Natural Language Processing (NLP)
1.  **Sentiment Analysis (`sentiment-analysis`)**:
    - Classify text as `POSITIVE` or `NEGATIVE`.
    - Demonstrates using both a default model and a specialized, multi-lingual model (`nlptown/bert-base-multilingual-uncased-sentiment`) for more granular, star-based ratings.
2.  **Named Entity Recognition (`ner`)**:
    - Identify and categorize entities like persons, organizations, and locations within a block of text.
3.  **Question Answering (`question-answering`)**:
    - Extract an answer to a specific question from a given context or document.
4.  **Text Summarization (`summarization`)**:
    - Generate a concise summary of a longer piece of text, with controls for length.
5.  **Translation (`translation_en_to_fr`, `translation_en_to_lg`)**:
    - Translate text from one language to another. Includes examples using both a default model and a community-contributed model for a less common language (Luganda).
6.  **Zero-Shot Classification (`zero-shot-classification`)**:
    - Classify text into custom categories on-the-fly, without needing a model pre-trained specifically on those labels.
7.  **Text Generation (`text-generation`)**:
    - Generate creative and coherent text based on an initial prompt, powered by a generative language model.

### Computer Vision
8.  **Image Generation (`text-to-image`)**:
    - Create stunning images from a text prompt using a diffusion model (`stabilityai/sdxl-turbo`) via the `diffusers` library pipeline.

### Audio
9.  **Audio Generation (`text-to-speech`)**:
    - Synthesize human-like speech from text using a Text-to-Speech (TTS) model (`microsoft/speecht5_tts`), including the ability to use different speaker embeddings.

---

## 🛠️ Getting Started

Follow these steps to set up your environment and run the notebook. This project is optimized for Google Colab to leverage its free T4 GPU resources.

### Prerequisites
- A Google Account (for Google Colab).
- A **Hugging Face Account** (create a free account at [huggingface.co](https://huggingface.co)).

### Setup and Installation

1.  **Open in Google Colab**:
    Click the "Open in Colab" badge at the top of this README to launch the notebook directly in your browser.

2.  **Configure your Hugging Face API Token**:
    This is a crucial step for accessing models from the Hugging Face Hub.
    - Go to your Hugging Face profile: **Settings -> Access Tokens**.
    - Create a new token. Make sure to give it **`write` permissions**.
    - In your Colab notebook, click the **key icon (🔑) on the left sidebar** to open "Secrets".
    - Create a new secret named `HF_TOKEN` and paste your Hugging Face access token into the value field.
    - **Enable the "Notebook access"** toggle for this secret.

3.  **Connect to a GPU Runtime**:
    To run the models efficiently, you need a GPU.
    - In Colab, go to **Runtime -> Change runtime type**.
    - Select **T4 GPU** from the "Hardware accelerator" dropdown menu and click "Save".
    - You can verify your GPU connection by running the `!nvidia-smi` cell in the notebook. It should show a "Tesla T4".

4.  **Install Dependencies**:
    The first code cell in the notebook handles the necessary `pip` installations. Simply run this cell to install the required libraries.

    ```bash
    !pip install -q --upgrade datasets transformers diffusers soundfile
    ```

### Running the Notebook
Once the setup is complete, you can run the cells sequentially from top to bottom to see each pipeline in action.

---

## 💡 Important Tips for Colab Users

This notebook includes critical advice for a smooth experience in Google Colab:

- **Ignore Safe Warnings**: Data science libraries often produce warnings. These can usually be ignored unless an actual error occurs.
- **Handling the "CUDA is required" Error**: This misleading error often appears when your Colab runtime is reset. The solution is **not** to change package versions. Instead:
    1.  Go to the **Kernel menu -> Disconnect and delete runtime**.
    2.  Reload the page and go to **Edit menu -> Clear All Outputs**.
    3.  Reconnect to a T4 GPU runtime.
    4.  Re-run all cells from the top, starting with the `pip install` command.

## 📚 Further Exploration

The possibilities with pipelines are vast. For a complete and up-to-date list of all available tasks, refer to the official Hugging Face documentation:

- **Transformers Pipelines**: [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- **Diffusers Pipelines**: [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/overview)
