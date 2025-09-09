# Model Quantization with llama.cpp

This guide provides instructions for setting up `llama.cpp` on Windows, converting Hugging Face models to the GGUF format, and performing various types of quantization, including standard, imatrix, and VLM quantization.

## Prerequisites

1.  A Conda environment.
2.  CMake: `conda install -c conda-forge cmake -y`
3.  Visual Studio 2022 with C++ development tools.
4.  CUDA Toolkit (if using GPU): `conda install cuda-toolkit=12.6 cuda-nvcc=12.6 -c nvidia -y`
5.  SentencePiece: `conda install -c conda-forge sentencepiece -y`
6.  Curl: `conda install -c conda-forge libcurl -y`
7.  Copy CUDA files from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions` to `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations`.

## Setup `llama.cpp` on Windows

First, clone the repository:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### CPU Build

1.  Create a build directory:
    ```bash
    mkdir build-gpu
    cd build-gpu
    ```
2.  Configure with CMake:
    ```bash
    cmake .. -G "Visual Studio 17 2022" -A x64
    ```
3.  Build the project:
    ```bash
    cmake --build . --config Release
    ```
4.  Return to the `llama.cpp` root directory:
    ```bash
    cd ..
    ```

### GPU Build (CUDA)

1.  Configure with CMake, enabling CUDA and specifying the toolkit location:
    ```bash
    cmake -B build-gpu -DGGML_CUDA=ON
    # If needed, specify CUDA location explicitly
    cmake -B build-gpu -DGGML_CUDA=ON -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    ```
2.  Build the project:
    ```bash
    cmake --build build-gpu --config Release
    ```

## Model Conversion and Quantization

### Converting HF Model to GGUF Format

1.  Download the desired Hugging Face model (e.g., using the `download_model_LLM.py` script).
2.  Navigate to the `llama.cpp` directory.
3.  Convert the model to GGUF format:
    ```bash
    python convert_hf_to_gguf.py --outtype f16 --outfile ../ggufs/Mistral-7B-Instruct-v0.3.gguf ../models/Mistral-7B-Instruct-v0.3
    ```

### Standard Quantization

1.  Navigate to the release binaries directory:
    ```bash
    cd build-gpu\bin\Release
    ```
2.  Run the quantization tool:
    ```bash
    .\llama-quantize.exe ..\..\..\..\ggufs\Mistral-7B-Instruct-v0.3.gguf ..\..\..\..\ggufs\Mistral-7B-Instruct-v0.3.Q4_K.gguf Q4_K
    ```

### Imatrix Quantization

1.  Generate the importance matrix:
    ```bash
    .\llama-imatrix.exe -m ..\..\..\..\..\ggufs\mistral-7b-instruct-v0.3.gguf -f ..\..\..\..\..\wiki.train.raw -o ..\..\..\..\..\ggufs\imatrix\mistral-7b-instruct-v0.3.imatrix.gguf --chunks 2000 --ctx-size 512 -ngl 80
    ```
2.  Quantize the model using the importance matrix:
    ```bash
    .\llama-quantize.exe --imatrix ..\..\..\..\..\ggufs\imatrix\mistral-7b-instruct-v0.3.imatrix.gguf ..\..\..\..\..\ggufs\mistral-7b-instruct-v0.3.gguf ..\..\..\..\..\ggufs\imatrix\mistral-7b-instruct-v0.3-Q4_K_M.gguf Q4_K_M
    ```

## Vision Language Model (VLM) - Qwen2.5VL

### VLM Setup

1.  Install required Python packages:
    ```bash
    pip install torch torchvision transformers accelerate qwen-vl-utils
    pip install git+https://github.com/huggingface/transformers
    ```
2.  Download the VLM model (e.g., using `download_model_VLM.py`).
3.  Build `llama.cpp` if you haven't already.
4.  Convert the model and generate the multimodal projector:
    ```bash
    # Convert the main model
    python convert_hf_to_gguf.py --outfile ../ggufs/qwen2.5-vl-7b.gguf --outtype f16 ../models/Qwen2.5-VL-7B-Instruct

    # Generate the multimodal projector
    python convert_hf_to_gguf.py --outfile ../ggufs/mmproj-qwen2.5-vl-7b.gguf --outtype f16 --mmproj ../models/Qwen2.5-VL-7B-Instruct
    ```
5.  Quantize the VLM:
    ```bash
    ./build-gpu/bin/Release/llama-quantize.exe ../ggufs/qwen2.5-vl-7b.gguf ../ggufs/qwen2.5-vl-7b-q4_k.gguf q4_k
    ```

### Running the VLM

-   **Single-turn query with an image:**
    ```bash
    ./build-gpu/bin/Release/llama-mtmd-cli.exe -m ../ggufs/qwen2.5-vl-7b-q4_k.gguf --mmproj ../ggufs/mmproj-qwen2.5-vl-7b.gguf --image path/to/your/image.jpg -p "What can you see in this image?"
    ```
-   **Interactive conversation mode:**
    ```bash
    ./build-gpu/bin/Release/llama-mtmd-cli.exe -m ../ggufs/qwen2.5-vl-7b-q4_k.gguf --mmproj ../ggufs/mmproj-qwen2.5-vl-7b.gguf -c 4096 --temp 0.7
    ```

## Quantization Formats

`llama.cpp` supports a wide range of quantization formats. You can check the `instruction.txt` file for a comprehensive list of basic, K-quantization, I-quantization, and special formats.

## Supported Models

To see a list of models supported by the `convert_hf_to_gguf.py` script, run:
```bash
python .\convert_hf_to_gguf.py --print-supported-models
```
The `instruction.txt` file also contains a list of supported TEXT and MMPROJ models.
