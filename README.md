# Analysis of Numerical Instability in Large Language Models

This repository contains the code and results for a series of experiments investigating the numerical instability of Large Language Models (LLMs) when subjected to small, near-rounding-error perturbations. The project analyzes how these minute changes in input embeddings can propagate through the model, leading to significant alterations in hidden states, logits, and ultimately, the predicted tokens.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
  - [Experiment 1: Layer-wise SVD Perturbation Analysis](#experiment-1-layer-wise-svd-perturbation-analysis)
  - [Experiment 2: Average Lipschitz Constant Computation](#experiment-2-average-lipschitz-constant-computation)
  - [Experiment 3: FP32 Precision Evaluation](#experiment-3-fp32-precision-evaluation)
  - [Experiment 4: GPU Output Comparison and Divergence Analysis](#experiment-4-gpu-output-comparison-and-divergence-analysis)
  - [Experiment 5: Layer-wise and Module-wise Lipschitz Analysis](#experiment-5-layer-wise-and-module-wise-lipschitz-analysis)
  - [Experiment 6: Optimization for Equal Logits and Token Flip Analysis](#experiment-6-optimization-for-equal-logits-and-token-flip-analysis)
  - [Experiment 7: Logit Difference Heatmaps and Randomness Quantification](#experiment-7-logit-difference-heatmaps-and-randomness-quantification)
  - [Experiment 8: Submodule Output Analysis](#experiment-8-submodule-output-analysis)
  - [Experiment 9: Weight Matrix Analysis](#experiment-9-weight-matrix-analysis)
  - [Experiment 10: PyTorch Determinism and Perturbation Analysis](#experiment-10-pytorch-determinism-and-perturbation-analysis)
  - [Experiment 11: Singular Vector Rotation Analysis](#experiment-11-singular-vector-rotation-analysis)
  - [Experiment 12: Polar Stability Boundary Analysis](#experiment-12-polar-stability-boundary-analysis)
  - [Experiment 13: Singular Vector Stability Analysis](#experiment-13-singular-vector-stability-analysis)
  - [Experiment 14: Lipschitz Constants via Small Steps Analysis](#experiment-14-lipschitz-constants-via-small-steps-analysis)
  - [Experiment 15: Stability Boundary L2 Distance Analysis](#experiment-15-stability-boundary-l2-distance-analysis)
  - [Additional Analysis: PDF Word-by-Word Prediction](#additional-analysis-pdf-word-by-word-prediction)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/llm_rounding_error_instability.git
    cd llm_rounding_error_instability
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

```
├── src/                        # Source code for all experiments
│   ├── exp1_layerwise_svd_perturbation_analysis.py
│   ├── exp2_average_lipschitz_constant_computation.py
│   ├── exp2_v2_average_lipschitz_constant_computation.py
│   ├── exp3_fp32_precision_evaluation.py
│   ├── exp4_part1_gpu_output_generation.py
│   ├── exp4_part2_gpu_output_comparison.py
│   ├── exp4_part3_model_output_divergence_analysis.py
│   ├── exp4_part4_internal_representation_plotting.py
│   ├── exp4_part5_embedding_perturbation_until_divergence.py
│   ├── exp4_part6_embedding_similarity_comparison.py
│   ├── exp5_layerwise_modulewise_lipschitz_analysis.py
│   ├── exp6_part1_geometric_approach_for_equal_logits.py
│   ├── exp6_part1_optimization_for_equal_probabilities.py
│   ├── exp6_part2_token_flip_causation_analysis.py
│   ├── exp7_part1_logit_difference_heatmap_generation.py
│   ├── exp7_part2_zoomed_in_logit_heatmap_plot.py
│   ├── exp7_part3_output_randomness_quantification.py
│   ├── pdf_word_by_word_prediction.py
│   ├── utils.py
│   └── ...
├── notebooks/                  # Jupyter notebooks for analysis and visualization
├── results/                    # Directory for storing experiment results
├── data/                       # Data files (e.g., PDFs for analysis)
├── docs/                       # Documentation
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Scripts

This section provides an overview of the main Python scripts in the `src` directory that are not part of a numbered experiment series.

### `src/llama_model_perturbation_test.py`

- **Purpose:** Tests the numerical instability of Llama models by applying small perturbations to input embeddings and measuring the impact on output predictions. It pays special attention to the final RMSNorm layer.
- **How to run:**
  ```bash
  python src/llama_model_perturbation_test.py
  ```
- **Inputs:**
  - `--model_path`: (Optional) Path to the Llama model. Defaults to `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`.
  - The script has a hardcoded list of `test_cases` (prompts) that it uses for the analysis.
- **Output:**
  - The script prints a detailed analysis to standard output.
  - It saves the results in CSV files, one for each test case (`llama_perturbation_test_case_1.csv`, etc.) and a combined analysis file (`llama_perturbation_combined_analysis.csv`). If saving to CSV fails, it saves to pickle files.

### `src/archive_lipschitz_constant_computation.py`

- **Purpose:** This script computes the Lipschitz constant of the Llama model by analyzing the Jacobian of the transformation from input embeddings to the final hidden states. It uses Singular Value Decomposition (SVD) to find the largest singular value, which corresponds to the Lipschitz constant.
- **How to run:**
  ```bash
  python src/archive_lipschitz_constant_computation.py
  ```
- **Inputs:**
  - `--model_path`: (Optional) Path to the Llama model. Defaults to `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`.
  - The script has a hardcoded list of `test_inputs` (prompts) that it uses for the analysis.
- **Output:**
  - The script prints a detailed analysis to standard output.
  - It saves the results for each test case into a CSV file named `lipschitz_analysis_test_<test_case_number>_complete.csv`.

### `src/utils.py`

- **Purpose:** Provides a set of utility functions for loading models, generating text, and extracting representations and logits. It is a core component used by many other scripts in the project.
- **How to run:** This file is not meant to be run directly as a script. It's a library to be imported by other scripts.

### `src/llama_model_lipschitz_computation_part2.py`

- **Purpose:** Computes the Lipschitz constant of the Llama model and analyzes rounding errors by saving hidden states and logits for perturbed and unperturbed inputs.
- **How to run:**
  ```bash
  python src/llama_model_lipschitz_computation_part2.py
  ```
- **Inputs:**
  - `--model_path`: (Optional) Path to the Llama model. Defaults to `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`.
  - The script has a hardcoded `input_text`.
  - `--output_dir`: (Optional) Directory to save the output files. Defaults to `./rounding_error_analysis`.
- **Output:**
  - The script prints a detailed analysis to standard output.
  - It saves the results in a text file `rounding_error_analysis.txt` and several `.npy` and `.npz` files in the specified output directory.

### `src/pdf_word_by_word_prediction.py`

- **Purpose:** This script analyzes a PDF document word by word, predicting the next word based on the preceding context and comparing it with the actual word in the document. It calculates various metrics like logits, probabilities, and cosine similarities.
- **How to run:**
  ```bash
  python src/pdf_word_by_word_prediction.py
  ```
- **Inputs:** The input PDF, model, and output CSV paths are hardcoded in the script. To change them, the script itself needs to be edited.
  - `PDF_PATH`: The path to the PDF file to be analyzed.
  - `MODEL_PATH`: The path to the language model.
  - `OUTPUT_CSV`: The path to save the output CSV file.
- **Output:**
  - The script prints a detailed analysis to standard output.
  - It saves the results in a CSV file specified by `OUTPUT_CSV`. It also saves checkpoint files.

### `src/gpt_oss_lipschitz_const.py`

- **Purpose:** Computes the Lipschitz constant of the GPT-OSS model using Jacobian analysis and SVD. This version is specifically designed to run on the CPU to avoid GPU memory issues.
- **How to run:**
  ```bash
  python src/gpt_oss_lipschitz_const.py
  ```
- **Inputs:** The script has a hardcoded list of `test_inputs` (prompts) that it uses for the analysis.
- **Output:**
  - The script prints a detailed analysis to standard output.
  - It saves the results for each test case into a CSV file named `gpt_oss_lipschitz_test_<test_case_number>_complete.csv`.

### `src/gpt_model_perturbation_test.py`

- **Purpose:** This script tests the numerical instability of GPT models by applying small perturbations to the input embeddings and observing the changes in the model's predictions.
- **How to run:** The user needs to add a `if __name__ == "__main__":` block to run this script and can redirect the output to a file.
  ```python
  if __name__ == "__main__":
      test_input = "The capital of France is"
      results_df = test_perturbation_effects(model, tokenizer, test_input)
      print(results_df)
  ```
  ```bash
  python src/gpt_model_perturbation_test.py > gpt_model_perturbation_test_output.txt
  ```
- **Inputs:**
  - The script hardcodes the model `openai/gpt-oss-20b`.
  - The `test_perturbation_effects` function takes an `input_text` argument.
- **Output:**
  - The script prints the results of the perturbation tests to standard output.
  - It returns a pandas DataFrame with the detailed results, but it does not save this DataFrame to a file by default.

## Experiments

### Experiment 1: Layer-wise SVD Perturbation Analysis

This experiment analyzes how perturbations in the input embedding affect the hidden states of the model, with a focus on layer-wise analysis and the impact of floating-point precision.

#### `src/exp1_cpu_fp64_analysis.py`

- **Purpose:** Analyzes the effect of small perturbations on the layer 0 hidden state using `float64` precision on the CPU. This provides a high-precision baseline for comparison.
- **How to run:**
  ```bash
  python src/exp1_cpu_fp64_analysis.py
  ```
- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text (hardcoded): "The capital of France is"
- **Output:**
  - Prints progress to standard output.
  - Saves a plot to `../results/exp1_layer0_count_and_maxdiff_vs_jump_cpu_fp64.pdf` and `.png`.
  - Saves raw data to `../results/exp1_layer0_results_cpu_fp64.json` and `.csv`.

#### `src/exp1_layer0_jump_analysis.py`

- **Purpose:** Similar to the `fp64` analysis, but uses `float32` precision and runs on the GPU (if available). It analyzes the effect of perturbations on the layer 0 output.
- **How to run:**
  ```bash
  python src/exp1_layer0_jump_analysis.py
  ```
- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text (hardcoded): "The capital of France is"
- **Output:**
  - Prints progress to standard output.
  - Saves a plot to `exp1_layer0_jump_analysis.pdf`.

#### `src/exp1_layerwise_svd_perturbation_analysis.py`

- **Purpose:** A comprehensive analysis of how perturbations along singular vector directions propagate through all layers of the model. It helps in understanding how instability propagates and which layers are most sensitive.
- **How to run:**
  ```bash
  python src/exp1_layerwise_svd_perturbation_analysis.py
  ```
- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text (hardcoded): "The capital of France is"
- **Output:**
  - Prints detailed statistics to standard output.
  - Creates a timestamped directory in `../results` (e.g., `../results/exp1_2025-12-17_14-11-53/`).
  - Saves top 5 singular vectors and values as `.npy` files.
  - For each perturbation "jump", it saves:
    - A `.npz` file with raw layer representations and comparison results.
    - A `.pdf` plot comparing layer-wise representations.
    - A `.pdf` heatmap of differences.

#### `src/exp1_plot_layer0_maxdiff_vs_jump.py`

- **Purpose:** Analyzes the effect of small perturbations on the first layer's hidden state using `float32` precision on a GPU. This script is very similar to `exp1_layer0_jump_analysis.py` but also saves the results in JSON and CSV formats.
- **How to run:**
  ```bash
  python src/exp1_plot_layer0_maxdiff_vs_jump.py
  ```
- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text (hardcoded): "The capital of France is"
- **Output:**
  - Prints progress to standard output.
  - Saves a plot to `../results/exp1_layer0_count_and_maxdiff_vs_jump.pdf` and `.png`.
  - Saves raw data to `../results/exp1_layer0_results.json` and `.csv`.

### Experiment 2: Average Lipschitz Constant Computation

This experiment estimates the local Lipschitz constant to quantify the model's sensitivity to input changes. It consists of three scripts, each offering a different version of the analysis.

#### `src/exp2_average_lipschitz_constant_computation.py`

- **Purpose:** This is the original script to compute the average Lipschitz constant across multiple prompts. It measures the typical Lipschitz constant of the Llama model by computing the Jacobian and SVD for different prompts and analyzing the distribution of the largest singular value.
- **How to run:**
  ```bash
  python src/exp2_average_lipschitz_constant_computation.py
  ```
- **Inputs:** A list of text prompts hardcoded in the script.
- **Output:**
  - A timestamped directory in `results/` like `exp2_YYYY-MM-DD_HH-MM-SS/`.
  - A CSV file with Lipschitz constants for each prompt.
  - Statistical summary (mean, std, min, max, median).
  - Distribution plots of Lipschitz constants.

#### `src/exp2_v2_average_lipschitz_constant_computation.py`

- **Purpose:** This is a revised version of the previous script. It may contain bug fixes or methodological improvements. It also computes the average Lipschitz constant across multiple prompts. A notable change is in the `compute_lipschitz_along_direction_with_noise` function where the noise is applied differently.
- **How to run:**
  ```bash
  python src/exp2_v2_average_lipschitz_constant_computation.py > exp2_avg_lipschitz_v2.txt 2>&1
  ```
  The script prints a lot of information to the console, so it's recommended to redirect the output to a file.
- **Inputs:** A list of text prompts hardcoded in the script.
- **Output:**
  - A timestamped directory in `results/` like `exp2_v2_YYYY-MM-DD_HH-MM-SS/`.
  - A CSV file with Lipschitz constants for each prompt.
  - Statistical summary.
  - Distribution plots.

#### `src/exp2_v3_average_lipschitz_constant_computation.py`

- **Purpose:** This script computes the Lipschitz constant with flexible precision (`bfloat16`, `float32`, `float64`). It automatically selects the device (CPU for `float64`, GPU otherwise). It computes the Jacobian and performs SVD to find the Lipschitz constant.
- **How to run:**
  First, set the `PRECISION` variable in the script to `torch.float64`, `torch.float32`, or `torch.bfloat16`.
  ```bash
  python src/exp2_v3_average_lipschitz_constant_computation.py
  ```
- **Inputs:** A list of text prompts hardcoded in the script. The precision is set via the `PRECISION` variable in the script.
- **Output:**
  - A timestamped directory in `results/` like `llama_lipschitz_<precision>_YYYY-MM-DD_HH-MM-SS/`.
  - A CSV file with a complete analysis for each test case.

### Experiment 3: FP32 Precision Evaluation

- **Purpose:** This experiment specifically focuses on analyzing model behavior and Lipschitz constants using `float32` precision. It investigates the effect of `float32` precision on perturbations and model behavior, and whether it provides sufficient numerical precision. It computes the Jacobian, performs SVD, and then analyzes the effect of perturbations on the embedding values at different precision levels.
- **Script:** [`src/exp3_fp32_precision_evaluation.py`](./src/exp3_fp32_precision_evaluation.py)
- **How to run:**
  ```bash
  python src/exp3_fp32_precision_evaluation.py > exp3_fp32_precision_evaluation.txt
  ```
  It is recommended to redirect the output to a file to capture the summary statistics.
- **Inputs:**
  - The script uses a hardcoded input text: "The capital of France is".
- **Output:**
  - A timestamped directory in `results/` like `exp3_YYYY-MM-DD_HH-MM-SS/`.
  - A CSV file `embedding_perturbation_analysis.csv` containing a detailed analysis of the embedding perturbations.
  - The script prints a summary to standard output, showing the number of dimensions that changed at different precision levels for various epsilon values.

### Experiment 4: GPU Output Comparison and Divergence Analysis

This experiment is designed to determine if running the same language model with the same inputs on different GPU hardware can lead to different outputs. It's a crucial test for hardware-specific numerical instability. The experiment is broken down into several parts, each handled by a separate script. They are intended to be run in sequence.

#### Part 1: `src/exp4_part1_gpu_output_generation.py`

- **Purpose:** Generates text token-by-token from a language model for a predefined set of questions. At each step, it saves detailed information, including hidden states, logits, and probabilities. This script should be run on each GPU you want to compare.
- **How to run:**

  ```bash
  # Run on your first GPU
  python src/exp4_part1_gpu_output_generation.py

  # Run on your second GPU
  python src/exp4_part1_gpu_output_generation.py
  ```

  You can also use the `--inspect_activations` flag to enable hooks that print the mean of MLP SiLU activations during generation.

- **Inputs:**
  - A Llama-3.1-8B-Instruct model (the path is hardcoded in the script).
  - A predefined list of 10 questions (5 factual, 5 designed to induce hallucinations).
- **Output Structure:**
  - Creates a timestamped directory in `../results/`, e.g., `../results/exp4_2025-12-18_10-00-00/`.
  - Inside this directory, it creates subdirectories for each question (`question_01/`, `question_02/`, etc.).
  - Each question subdirectory contains:
    - `representations.npy`: The final normalized hidden states for each generated token.
    - `top_10_logits.npy`: The top 10 logits for each generated token.
    - `top_10_probs.npy`: The corresponding probabilities for the top 10 logits.
    - `words.json`: The generated words, top 5 predicted words, and token IDs.
    - `metadata.json`: The input prompt and other metadata.
- **Saving Console Output:**
  ```bash
  python src/exp4_part1_gpu_output_generation.py > exp4_part1_gpu1.log
  ```

#### Part 2: `src/exp4_part2_gpu_output_comparison.py`

- **Purpose:** Compares the outputs generated by Part 1 from two different runs (e.g., from two different GPUs). It identifies if and where the generated text diverges and quantifies the differences in hidden states and logits.
- **How to run:**
  ```bash
  python src/exp4_part2_gpu_output_comparison.py <path_to_gpu1_results> <path_to_gpu2_results> --num-questions 10
  ```
- **Inputs:**
  - Two folder paths pointing to the output directories from Part 1.
- **Output Structure:**
  - Prints a detailed comparison summary to the console.
  - Saves a `comparison_results.json` file in a new timestamped directory in `../results/`, containing a detailed analysis of divergence points, representation differences (cosine similarity, L2 distance), and logit correlations.
- **Saving Console Output:**
  ```bash
  python src/exp4_part2_gpu_output_comparison.py <folder1> <folder2> > exp4_part2_comparison_summary.log
  ```

#### Part 3: `src/exp4_part3_model_output_divergence_analysis.py`

- **Purpose:** Performs a fine-grained, dimension-level analysis of the hidden state representations at the exact point of divergence identified in Part 2. It counts how many dimensions have changed and by how much.
- **How to run:**
  ```bash
  python src/exp4_part3_model_output_divergence_analysis.py <path_to_repr1.npy> <path_to_repr2.npy> <divergence_index>
  ```
- **Inputs:**
  - Paths to the `representations.npy` files from the two runs you are comparing.
  - The integer index where the divergence occurred (found in Part 2).
- **Output Structure:**
  - Prints a report to the console detailing the percentage of dimensions changed at various precision thresholds (e.g., >1e-3, >1e-5), along with other statistics.
- **Saving Console Output:**
  ```bash
  python src/exp4_part3_model_output_divergence_analysis.py <file1> <file2> <index> > exp4_part3_divergence_analysis.log
  ```

#### Part 4: `src/exp4_part4_internal_representation_plotting.py`

- **Purpose:** Visualizes how the hidden state representations from the two runs drift apart over the sequence of generated tokens by plotting their cosine similarity and L2 distance.
- **How to run:**
  ```bash
  python src/exp4_part4_internal_representation_plotting.py <path_to_repr1.npy> <path_to_repr2.npy> [output.png] [divergence_index]
  ```
- **Inputs:**
  - Paths to the `representations.npy` files from the two runs.
  - (Optional) An output filename for the plot.
  - (Optional) The divergence index to mark on the plot.
- **Output Structure:**
  - Saves a PNG image file in a new timestamped directory in `../results/` with two subplots: cosine similarity vs. token position and L2 distance vs. token position.
  - Prints summary statistics to the console.
- **Saving Console Output:**
  ```bash
  python src/exp4_part4_internal_representation_plotting.py <file1> <file2> > exp4_part4_summary.log
  ```

#### Part 5: `src/exp4_part5_embedding_perturbation_until_divergence.py`

- **Purpose:** This script helps determine if the divergence originates from the very first step (the embedding layer) or later in the model. It extracts the input embeddings for the token sequence up to the point of divergence. The idea is to run this on two GPUs and then compare the saved embedding files in Part 6.
- **How to run:**
  ```bash
  python src/exp4_part5_embedding_perturbation_until_divergence.py <results_folder> <divergence_source.json> --model-path <path_to_model>
  ```
- **Inputs:**
  - The results folder from a Part 1 run.
  - A JSON file containing the divergence indices (can be the output from Part 2).
  - The path to the language model.
- **Output Structure:**
  - Creates a new directory containing the extracted input embeddings (`question_XX_embeddings.npy`) for each question that had a divergence.
- **Saving Console Output:**
  ```bash
  python src/exp4_part5_embedding_perturbation_until_divergence.py [args] > exp4_part5_log.txt
  ```

#### Part 6: `src/exp4_part6_embedding_similarity_comparison.py`

- **Purpose:** Compares the embedding files generated in Part 5 from two different runs to check if they are identical. This is the final check to see if the hardware differences appear during the embedding lookup or later during computation.
- **How to run:**
  ```bash
  python src/exp4_part6_embedding_similarity_comparison.py <path_to_embeddings_folder1> <path_to_embeddings_folder2>
  ```
- **Inputs:**
  - Two folders containing the `.npy` embedding files from Part 5.
- **Output Structure:**
  - Prints a detailed summary to the console for each question, showing if token embeddings differ.
  - Saves a JSON file with the comparison results.
  - Generates plots visualizing the similarity of the embeddings for each token.
- **Saving Console Output:**
  ```bash
  python src/exp4_part6_embedding_similarity_comparison.py [args] > exp4_part6_summary.log
  ```

#### Part 7: Logit and Angle Analysis

These scripts provide a deeper look into _why_ the model's predictions diverge.

- **`src/exp4_part7_compare_logits.py`**

  - **Purpose:** Focuses on the top-5 logits at the exact point of divergence, showing a detailed table comparing the predicted words, their logits, and probabilities from each run.
  - **How to run:** You must edit the script to provide the paths to the two results folders from Part 1, then run `python src/exp4_part7_compare_logits.py`.
  - **Output:** Prints a detailed comparison report and a summary DataFrame to the console.
  - **Saving Console Output:** `python src/exp4_part7_compare_logits.py > exp4_part7_logit_comparison.log`

- **`src/exp4_part7_LPT_angle_dist_with_unembedding_vectors.py`**
  - **Purpose:** Performs a more theoretical analysis by calculating the angle between the final hidden state and the unembedding vectors for the top predicted tokens. This helps understand the geometry of the model's decision space.
  - **How to run:** You must edit the script to provide the model path and the path to a results folder from Part 1, then run `python src/exp4_part7_LPT_angle_dist_with_unembedding_vectors.py`.
  - **Output:** Creates a new results directory containing plots (histograms, box plots, etc.) of the angle distributions and a JSON file with detailed statistics.
  - **Saving Console Output:** `python src/exp4_part7_LPT_angle_dist_with_unembedding_vectors.py > exp4_part7_angle_analysis_summary.log`

### Experiment 5: Layer-wise and Module-wise Lipschitz Analysis (`exp5_layerwise_modulewise_lipschitz_analysis.py`)

This experiment performs a fine-grained analysis of model instability by examining each layer and its submodules (e.g., self-attention, MLP). It computes layer-specific Jacobians and their Singular Value Decompositions (SVDs) to identify which layers and submodules are most sensitive to perturbations in the input.

**Description:**

The script loads a Llama-3.1-8B-Instruct model and applies small perturbations to the input embeddings. The perturbations are directed along the singular vectors of the model's Jacobian matrix. It then tracks the propagation of these perturbations through the submodules of a specified layer. This allows for a detailed, layer-by-layer view of instability.

**How to Run:**

The script is configured and run directly from the command line. The main parameters are set within the `if __name__ == "__main__":` block of the script.

Key parameters to configure inside the script:

- `layer_idx`: The index of the layer to analyze.
- `singular_idx`: The index of the singular vector to use for the perturbation direction.
- `e1`, `step_size`, `jumps`: Parameters controlling the magnitude of the perturbations.
- `text`: The input text for the model.

To run the script:

```bash
python src/exp5_layerwise_modulewise_lipschitz_analysis.py
```

To save the detailed standard output to a log file, redirect the output:

```bash
python src/exp5_layerwise_modulewise_lipschitz_analysis.py > exp5_output.log
```

**Input:**

- The script requires access to the Llama-3.1-8B-Instruct model weights. The path is hardcoded in the `load_model` function.
- The input text is hardcoded in the `if __name__ == "__main__":` block.

**Output Structure:**

The script generates a timestamped directory for each run, located at `results/exp5_YYYY-MM-DD_HH-MM-SS/`. This directory contains:

- **`.npy` files:**

  - `submodule_whole_model_svd_whole_model_top5_singular_vectors.npy`: The top 5 singular vectors for the entire model's Jacobian.
  - `submodule_whole_model_svd_whole_model_top5_singular_values.npy`: The corresponding top 5 singular values.

- **`.npz` file:**

  - `submodule_whole_model_svd_layer{...}.npz`: A compressed NumPy archive containing detailed results for the analyzed layer, including:
    - The outputs of each submodule for the original and perturbed inputs.
    - A dictionary of comparison metrics (e.g., L2 distance, cosine similarity).

- **`.pdf` file:**
  - `submodule_comparison_layer{...}.pdf`: A plot visualizing the differences in submodule outputs between the original and perturbed runs, helping to identify which submodules amplify the perturbation the most.

### Experiment 6: Optimization for Equal Logits and Token Flip Analysis

This experiment is divided into two main parts. The first part focuses on finding a precise input embedding that forces the model to produce equal logits (or probabilities) for its top two predicted next tokens. The second part uses this unstable "tipping point" to analyze what kind of small perturbations cause the model's prediction to flip from one token to the other.

#### `src/exp6_part1_geometric_approach_for_equal_logits.py`

- **Purpose:** This script uses a geometric approach to find an input embedding that makes the logits of the top two most likely next tokens equal.
- **How it works:** It's based on the idea that for two token unembeddings, `v1` and `v2`, if the final hidden state `h` is orthogonal to their difference (`v1` - `v2`), then the logits will be equal (`h · v1 = h · v2`). The script optimizes the input embedding to make `h · (v1 - v2)` as close to zero as possible.
- **How to run:**
  ```bash
  python src/exp6_part1_geometric_approach_for_equal_logits.py
  ```
- **Inputs:**
  - `MODEL_PATH` (hardcoded): The path to the language model.
  - `input_text` (hardcoded): The input text to the model.
- **Output:**
  - It creates a directory in `../results/` named `exp6_part1_geom_<timestamp>`.
  - Inside this directory, it saves three files:
    - `optimized_last_token_embedding_geometric.npy`: The optimized embedding for the last token.
    - `optimized_last_hidden_geometric.npy`: The final hidden state produced with the optimized embedding.
    - `optimized_state_geometric.pt`: Contains the full embeddings, the index of the last token, and the input text. This file is required by Part 2.

#### `src/exp6_part1_optimization_for_equal_probabilities.py`

- **Purpose:** This script optimizes the input embedding to make the probabilities of the top two most likely next tokens equal.
- **How it works:** It directly minimizes the squared difference between the logits of the top two tokens, `(L1 - L2)^2`. The model's weights are frozen, and only the input embeddings are trained.
- **How to run:**
  ```bash
  python src/exp6_part1_optimization_for_equal_probabilities.py
  ```
- **Inputs:**
  - `MODEL_PATH` (hardcoded): The path to the language model.
  - `input_text` (hardcoded): The input text to the model.
- **Output:**
  - It creates a directory in `../results/` named `exp6_part1_<timestamp>`.
  - Inside this directory, it saves three files:
    - `optimized_last_token_embedding.npy`: The optimized embedding for the last token.
    - `optimized_last_hidden.npy`: The final hidden state produced with the optimized embedding.
    - `optimized_state.pt`: Contains the full embeddings, the index of the last token, and the input text.

#### `src/exp6_part2_token_flip_causation_analysis.py`

- **Purpose:** This script analyzes why a model's token prediction "flips" from one token to another. It does this by perturbing the input embedding in specific, sensitive directions and observing when the prediction changes.
- **How to run:**
  1.  First, you must have run `src/exp6_part1_geometric_approach_for_equal_logits.py` to generate the `optimized_state_geometric.pt` file.
  2.  You must move or copy `optimized_state_geometric.pt` from its output directory (e.g., `../results/exp6_part1_geom_<timestamp>/`) into the `src` directory.
  3.  Then, run the script:
      ```bash
      python src/exp6_part2_token_flip_causation_analysis.py
      ```
- **Inputs:**
  - `MODEL_PATH` (hardcoded): The path to the language model.
  - `optimized_state_geometric.pt`: This file, generated by `exp6_part1_geometric_approach_for_equal_logits.py`, contains the starting embedding.
- **Output:**
  - It creates a directory in `../results/` named `exp6_part2_<timestamp>`.
  - Inside this directory, it saves a CSV file for each of the top 5 singular directions analyzed:
    - `token_flip_direction_<k>_geom_float32.csv`: This file logs the predicted token, its probability, and whether a flip occurred for different perturbation strengths.
  - The console output provides a summary of when the first token flip occurs for each direction. To capture this, you can redirect the output:
    ```bash
    python src/exp6_part2_token_flip_causation_analysis.py > ../results/exp6_part2_summary.log
    ```

### Experiment 7: Logit Difference Heatmaps and Randomness Quantification

This experiment investigates the decision boundary of the language model by creating 2D heatmaps of the logit difference between the top two token predictions. It perturbs an input embedding along two singular vector directions and observes the logit changes. The experiment is divided into four parts.

#### `src/exp7_part1_logit_difference_heatmap_generation.py`

- **Purpose:** Generates 2D heatmaps that visualize the model's decision boundary. It perturbs an input embedding along two singular vector directions and records the change in the logit difference between the top two candidate tokens.
- **How to run:**
  ```bash
  python src/exp7_part1_logit_difference_heatmap_generation.py
  ```
- **Inputs:**
  - `MODEL_PATH` (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`.
  - `input_text` (hardcoded): "The capital of France is".
- **Output:**
  - Creates a timestamped directory in `../results/` (e.g., `../results/exp7_2025-12-18_10-00-00/`).
  - Inside this directory, it saves three sets of files for different singular vector pairs:
    - `logit_diff_..._.png`: A color heatmap of the logit differences.
    - `logit_diff_..._bw.png`: A black and white heatmap showing the decision boundary.
    - `logit_diff_..._.npz`: A NumPy data file containing the raw grid data, which is used by the other `exp7` scripts.

#### `src/exp7_part2_zoomed_in_logit_heatmap_plot.py`

- **Purpose:** Takes the `.npz` files generated by `part1` and creates a "zoomed-in" view of a specific rectangular region of the heatmap. This allows for a more detailed inspection of the decision boundary.
- **How to run:**
  ```bash
  python src/exp7_part2_zoomed_in_logit_heatmap_plot.py <path_to_npz_file> --e1_start <val> --e1_end <val> --e2_start <val> --e2_end <val>
  ```
  Example:
  ```bash
  python src/exp7_part2_zoomed_in_logit_heatmap_plot.py ../results/exp7_2025-12-18_10-00-00/logit_diff_1st_2nd.npz --e1_start -5e-7 --e1_end 5e-7 --e2_start -5e-7 --e2_end 5e-7
  ```
- **Inputs:**
  - `filename`: The path to the `.npz` file from `part1`.
  - `--e1_start`, `--e1_end`: The `e1` perturbation range.
  - `--e2_start`, `--e2_end`: The `e2` perturbation range.
- **Output:**
  - Creates a timestamped directory in `../results/`.
  - Saves two image files for the specified zoomed region: `..._color.png` and `..._bw.png`.

#### `src/exp7_part3_output_randomness_quantification.py`

- **Purpose:** Performs a statistical analysis on the `.npz` files to quantify the "randomness" or "chaos" of the decision boundary.
- **How to run:**
  ```bash
  python src/exp7_part3_output_randomness_quantification.py <path_to_npz_file> --output <prefix> > ../results/exp7_part3_randomness_analysis.txt
  ```
  Example:
  ```bash
  python src/exp7_part3_output_randomness_quantification.py ../results/exp7_2025-12-18_10-00-00/logit_diff_1st_2nd.npz --output "RSV1and2" > ../results/exp7_2025-12-18_10-00-00/randomness_analysis.txt
  ```
- **Inputs:**
  - `filename`: The path to the `.npz` file from `part1`.
  - `--output` (optional): A prefix for the output plot file.
- **Output:**
  - Prints a detailed statistical analysis to standard output (which is redirected to a file in the command above).
  - Creates a timestamped directory in `../results/`.
  - Saves a PNG file named `{output_prefix}_analysis.png` containing several analysis plots.

#### `src/exp7_part4_enharge_white_areas.py`

- **Purpose:** Regenerates the black and white heatmaps from the `.npz` files with an option to "enlarge" the white areas to make the decision boundary more visible.
- **How to run:**
  - To process a single file:
    ```bash
    python src/exp7_part4_enharge_white_areas.py <path_to_npz_file> [enlarge_pixels]
    ```
  - To process all `.npz` files in a directory:
    `bash
python src/exp7_part4_enharge_white_areas.py <directory_path> [enlarge_pixels]
`
    Examples:
  ```bash
  python src/exp7_part4_enharge_white_areas.py ../results/exp7_2025-12-18_10-00-00/logit_diff_1st_2nd.npz 3
  python src/exp7_part4_enharge_white_areas.py ../results/exp7_2025-12-18_10-00-00/ 3
  ```
- **Inputs:**
  - A path to a single `.npz` file or a directory of them.
  - `enlarge_pixels` (optional): The amount to expand the white regions.
- **Output:**

  - For each `.npz` file, it saves a new `{npz_filename}_bw_enlarged.png` in the same directory.

### Experiment 8: Submodule Output Analysis

This set of experiments is designed to analyze the internal representations of the model at a submodule level.

#### Part 1: Submodule Output Generation

- **Description:** This script generates text token-by-token and saves the intermediate representations of specific submodules.
- **Input:** A list of questions embedded in the script.
- **Output Structure:**
  - A timestamped directory is created in the `results/` folder (e.g., `results/exp8_part1_YYYY-MM-DD_HH-MM-SS`).
  - Inside this directory, a subdirectory is created for each question (e.g., `question_01/`, `question_02/`).
  - Each question subdirectory contains:
    - `.npy` files for captured representations (e.g., `input_embeddings.npy`, `layer0_input_layernorm_outputs.npy`).
    - `words.json`: Contains the generated tokens and top-5 predictions.
    - `metadata.json`: Contains metadata about the generation process.
- **How to run:**
  ```bash
  python src/exp8_part1_submodule_output_generation.py
  ```
  The output is saved to a new directory in `results/`, so no redirection is needed.

#### Part 2: Divergence Analysis

- **Description:** This script compares all submodule outputs generated by `exp8_part1` from two different runs (e.g., from two different GPUs). It analyzes the outputs to pinpoint where and how they diverge.
- **Input:**
  1.  Two results directories from an `exp8_part1` run.
- **Output Structure:** A JSON file containing the detailed comparison results.
- **How to run:**
  ```bash
  python src/exp8_part2_divergence_analysis.py <folder1> <folder2> --output_file <output_file.json> --num_questions <num_questions>
  ```
  - `<folder1>`: Path to the first `exp8_part1` results directory.
  - `<folder2>`: Path to the second `exp8_part1` results directory.
  - `<output_file.json>`: The name of the file to save the analysis results.
  - `<num_questions>`: The number of questions to compare.

#### Part 3: Aggregate RMSNorm Impact Analysis

- **Description:** This script analyzes how the final RMSNorm affects alignment with top-5 unembedding vectors across all questions, focusing on divergence indices.
- **Input:**
  1.  A results directory from an `exp8_part1` run.
  2.  A `divergence_analysis.json` file from an `exp8_part2` run.
- **Output Structure:** A JSON file containing the aggregate analysis of the RMSNorm impact.
- **How to run:**
  ```bash
  python src/exp8_part3_aggregate_rmsnorm.py <result_dir> <divergence_file.json> --output_file <output_file.json>
  ```
  - `<result_dir>`: Path to the `exp8_part1` results directory.
  - `<divergence_file.json>`: Path to the `divergence_analysis.json` file from `exp8_part2`.
  - `<output_file.json>`: The name of the file to save the analysis results.

#### Part 4: Layer-by-Layer Divergence Localization

- **Description:** This script traces through the network layer-by-layer to identify WHERE the representation shift occurs that eventually leads to different token predictions.
- **Input:**
  1.  Two results directories from an `exp8_part1` run (one for each GPU).
  2.  A `divergence_analysis.json` file from an `exp8_part2` run.
- **Output Structure:** A JSON file containing the layer-by-layer divergence analysis.
- **How to run:**
  ```bash
  python src/exp8_part4_layer_divergence.py <gpu1_dir> <gpu2_dir> <divergence_file.json> --output_file <output_file.json>
  ```
  - `<gpu1_dir>`: Path to the first `exp8_part1` results directory.
  - `<gpu2_dir>`: Path to the second `exp8_part1` results directory.
  - `<divergence_file.json>`: Path to the `divergence_analysis.json` file from `exp8_part2`.
  - `<output_file.json>`: The name of the file to save the analysis results.

#### Part 5: Comprehensive Submodule Data Collection

- **Description:** This script captures the intermediate representations from **all submodules** (e.g., `input_layernorm`, `self_attn`, `mlp`, etc.) across **all layers** during token-by-token generation. It is a comprehensive data gathering tool.
- **How to run:**
  The script is run directly and does not take command-line arguments.
  ```bash
  python src/exp8_part5_more_submodule_all_layer.py
  ```
  To save the console output for logging:
  ```bash
  python src/exp8_part5_more_submodule_all_layer.py > exp8_part5_output.log
  ```
- **Input:**
  - `MODEL_PATH` (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`.
  - A hardcoded list of questions to generate text from.
- **Output Structure:**
  - Creates a timestamped directory in `../results`, e.g., `../results/exp8_part5_comprehensive_float32_YYYY-MM-DD_HH-MM-SS/`.
  - Inside, it creates a subdirectory for each question (e.g., `question_01/`).
  - Each question directory contains a large number of `.npy` files, one for each submodule at each layer (e.g., `layer0_input_layernorm.npy`, `layer15_mlp.npy`), along with `words.json` and `metadata.json`. This output is required by `exp8_part6` and `exp8_part8`.

#### Part 6: Comprehensive Layer-by-Layer Divergence Localization

- **Description:** This script traces through all submodules in all layers to identify the _exact point_ where the model's internal representation first "flips" to favor a different next token between two runs. It compares the dot product of the representation at every stage with the unembedding vectors of the top-2 candidate tokens.
- **How to run:**
  ```bash
  python src/exp8_part6_comprehensive_layer_divergence.py <gpu1_dir> <gpu2_dir> <divergence_file> --output_file <output.json>
  ```
  Example:
  ```bash
  python src/exp8_part6_comprehensive_layer_divergence.py \
      ../results/GPU1_exp8_part5_results/ \
      ../results/GPU2_exp8_part5_results/ \
      ../results/exp8_part2_divergence_analysis.json \
      --output_file exp8_part6_divergence_point.json --verbose
  ```
  To save the verbose console output:
  ```bash
  python src/exp8_part6_comprehensive_layer_divergence.py <args> > exp8_part6_summary.log
  ```
- **Input:**
  - `gpu1_dir`: Path to the `exp8_part5` results for the first run.
  - `gpu2_dir`: Path to the `exp8_part5` results for the second run.
  - `divergence_file`: The `divergence_analysis.json` file that identifies at which token the models diverged.
- **Output Structure:**
  - A JSON file (e.g., `exp8_part6_divergence_point.json`) containing a detailed breakdown of which token was favored at every stage of the network for both runs, and identifies the first point of disagreement.
  - Prints a summary to stdout, or a detailed stage-by-stage comparison if `--verbose` is used.

#### Part 7: GPU-Specific Dot Product Analysis

- **Description:** This script isolates the final dot product calculation to determine if hardware-specific numerics are the cause of divergence. It loads representations and computes the dot product with the unembedding matrix on the _current machine's GPU_, focusing only on the tokens at and right before the divergence point.
- **How to run:**
  This script should be run on each of the machines being compared.

  ```bash
  # On Machine 1
  python src/exp8_part7_gpu_specific_dotproduct.py path/to/machine1/part5_results/ \
      --divergence_json path/to/divergence.json \
      --output machine1_dot_products.json

  # On Machine 2
  python src/exp8_part7_gpu_specific_dotproduct.py path/to/machine2/part5_results/ \
      --divergence_json path/to/divergence.json \
      --output machine2_dot_products.json
  ```

  To save the console output:

  ```bash
  python src/exp8_part7_gpu_specific_dotproduct.py <args> > exp8_part7_machine1.log
  ```

- **Input:**
  - `results_dir`: Path to the `exp8_part5` results for the specific machine.
  - `--divergence_json`: The divergence analysis file.
- **Output Structure:**
  - A JSON file (e.g., `machine1_dot_products.json`) containing the dot product results for every submodule at the critical tokens, as calculated on that specific GPU.
  - Prints a detailed comparison table to stdout.

#### Part 8: Direct Representation Comparison

- **Description:** Performs a direct, value-by-value numerical comparison of the representations from two different runs. For a given token, it compares the activation vectors from every submodule and computes cosine similarity, L2 distance, percentage of changed values, and a highly detailed decimal precision analysis (up to 1e-15).
- **How to run:**
  ```bash
  python src/exp8_part8_rep_comparison_two_gpus.py <gpu1_dir> <gpu2_dir> \
      --divergence_file <divergence.json> --output <output.json> --verbose --plot
  ```
  Example:
  ```bash
  python src/exp8_part8_rep_comparison_two_gpus.py \
      ../results/GPU1_exp8_part5_results/ \
      ../results/GPU2_exp8_part5_results/ \
      --divergence_file ../results/divergence.json \
      --output gpu_comparison.json --verbose --plot
  ```
  To save the verbose console output:
  ```bash
  python src/exp8_part8_rep_comparison_two_gpus.py <args> > exp8_part8_summary.log
  ```
- **Input:**
  - `gpu1_dir`: Path to `exp8_part5` results for the first run.
  - `gpu2_dir`: Path to `exp8_part5` results for the second run.
  - `--divergence_file` (optional): If provided, the script analyzes the token just before divergence.
  - `--token_idx` (optional): A specific token index to analyze.
- **Output Structure:**
  - A detailed JSON file (e.g., `gpu_comparison.json`) with the full numerical comparison for all submodules.
  - If `--plot` is used, it generates several `.png` files visualizing the comparison with heatmaps and graphs.
  - Prints summary tables to stdout if `--verbose` is used.

### Experiment 9: Weight Matrix Analysis

This experiment focuses on analyzing the weight matrices of the model to understand their statistical properties and distributions.

#### `src/exp9_part1_weight_matrix_analysis.py`

- **Purpose:** This script performs a detailed analysis of the weight matrices of the attention projection layers (Q, K, V, O) in a Llama model. It computes various statistics like mean, standard deviation, norms, and distribution properties for each layer and projection type.
- **How to run:**
  ```bash
  python src/exp9_part1_weight_matrix_analysis.py > exp9_part1_output.log
  ```
- **Inputs:**
  - A Llama model, which is loaded from the path `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`. This path is hardcoded in the script.
- **Output Structure:**
  - The script creates a new directory in `../results` named `exp9_part1_weight_analysis_<timestamp>`.
  - Inside this directory, it saves:
    - **CSV files:** `weights_<projection>_statistics.csv` for each projection type (q_proj, k_proj, v_proj, o_proj).
    - **PDF plots:** Various plots visualizing the weight statistics across layers.
    - **NumPy file:** `weights_<projection>_raw.npz` for each projection, containing the raw weight matrices.

#### `src/exp9_part2_save_first_layer_weights.py`

- **Purpose:** This script saves the weights of all submodules in the first transformer layer of a Llama model to individual `.npy` files. It also prints some basic statistics for each weight matrix.
- **How to run:**
  ```bash
  python src/exp9_part2_save_first_layer_weights.py > exp9_part2_output.log
  ```
- **Inputs:**
  - A Llama model, which is loaded from the path `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`. This path is hardcoded in the script.
- **Output Structure:**
  - The script creates a directory named `first_layer_weights` in the `src/` directory.
  - Inside this directory, it saves the weights of each submodule of the first layer as a `.npy` file. The file names are in the format `layer0_<submodule_name>_weights.npy`.

### Experiment 10: PyTorch Determinism and Perturbation Analysis

This experiment investigates the determinism of PyTorch operations and analyzes the effect of small perturbations on model components. It is crucial for understanding the sources of non-determinism and numerical instability in deep learning models.

#### `src/exp10_part1_pytorch_determinism_test.py`

- **Purpose:** Tests the determinism of PyTorch tensor operations by running a chain of matrix multiplications with strict determinism settings. This script is used to generate a baseline tensor output.
- **How to run:**
  ```bash
  python src/exp10_part1_pytorch_determinism_test.py --output_file <output_tensor.pt> --device <device>
  ```
- **Inputs:**
  - `--output_file`: Path to save the output tensor. Default: `tensor_output.pt`.
  - `--seed`: Random seed. Default: `42`.
  - `--device`: Device to run on (e.g., `cuda`, `cpu`). Default: `cuda`.
- **Output:**
  - Saves a PyTorch tensor to the specified output file.

#### `src/exp10_part2_compare_tensors.py`

- **Purpose:** Compares two tensor files saved by `part1` to check if they are bit-for-bit identical. This is used to verify if running the same code on different hardware or at different times produces the exact same result.
- **How to run:**
  ```bash
  python src/exp10_part2_compare_tensors.py <tensor_file_1.pt> <tensor_file_2.pt>
  ```
- **Inputs:**
  - `file1`: Path to the first tensor file.
  - `file2`: Path to the second tensor file.
- **Output:**
  - Prints a message to standard output indicating whether the tensors are identical or different, along with statistics about the differences if they are not identical.

#### `src/exp10_part3_perturbation_effect.py`

- **Purpose:** Analyzes how a small perturbation applied to an initial vector is amplified as it passes through a sequence of 12 random matrix multiplications. This simulates the effect of depth in a neural network.
- **How to run:**
  ```bash
  python src/exp10_part3_perturbation_effect.py --dim 4096 --weight_scale 0.01 > exp10_part3_output.log
  ```
- **Inputs:**
  - `--dim`: Dimension of the square weight matrices. Default: `4096`.
  - `--weight_scale`: Scaling factor for the random weight matrices. Default: `0.01`.
  - `--device`: Device to run on. Default: `cuda` if available.
- **Output:**
  - Prints a detailed analysis to standard output, showing the magnification factor for a range of perturbation sizes (epsilons). It is recommended to redirect the output to a file.

#### `src/exp10_part4_embedding_perturbation.py`

- **Purpose:** Investigates the effect of a small perturbation on a real embedding vector from the Llama model. The perturbed embedding is passed through a single weight matrix (from the model's first layer) to measure the amplification.
- **How to run:**
  ```bash
  python src/exp10_part4_embedding_perturbation.py --weight_path <path_to_weights.npy> > exp10_part4_output.log
  ```
  To use the default weights, you must first run `src/exp9_part2_save_first_layer_weights.py`.
- **Inputs:**
  - `--weight_path`: Path to the `.npy` file for the weight matrix. Default: `first_layer_weights/layer0_self_attn_q_proj_weights.npy`.
  - `--text`: Input text to generate the embedding from. Default: "The capital of France is".
- **Output:**
  - Prints a detailed analysis to standard output, showing the magnification factor for a range of epsilon values. It is recommended to redirect the output to a file.

#### `src/exp10_part5_mlp_perturbation.py`

- **Purpose:** Extends the analysis to a full MLP block from the Llama model. It measures how a small perturbation on a real embedding vector is amplified after passing through the three matrix multiplications and the non-linear activation function (SiLU) of the MLP block.
- **How to run:**
  ```bash
  python src/exp10_part5_mlp_perturbation.py > exp10_part5_output.log
  ```
  To run this, you must first run `src/exp9_part2_save_first_layer_weights.py` to generate the required weight files.
- **Inputs:**
  - `--text`: Input text to generate the embedding from. Default: "The capital of France is".
- **Output:**
  - Prints a detailed analysis to standard output, showing the magnification factor of the perturbation through the MLP block for various epsilon values. It is recommended to redirect the output to a file.

### Experiment 11: Singular Vector Rotation Analysis

This experiment investigates how rotating the direction of a perturbation vector affects the model's hidden states.

#### `src/exp11_singular_vector_rotation_analysis.py`

- **Purpose:** This script investigates the effect of rotating a singular vector on the hidden states of a language model. It analyzes how the number and location of changes in the model's internal representations are affected by rotating the perturbation direction.
- **How to run:**
  ```bash
  python src/exp11_singular_vector_rotation_analysis.py
  ```
- **Input:**
  - The script takes several parameters, including perturbation magnitudes (`e1`, `step_size`, `jumps`), the singular vector index, and the input text. It loads a Llama model.
- **Output:**
  - The script generates a JSON file named `exp11_results.json` containing the analysis results.
  - The script prints its progress to standard output.
- **Command to save output:**
  ```bash
  python src/exp11_singular_vector_rotation_analysis.py > exp11_singular_vector_rotation_analysis.log
  ```

#### `src/exp11_part2_plot_results.py`

- **Purpose:** This script reads the rotation analysis results from the JSON file generated by `exp11_singular_vector_rotation_analysis.py` and creates a set of plots to visualize the findings.
- **How to run:**
  ```bash
  python src/exp11_part2_plot_results.py
  ```
- **Input:**
  - A JSON file named `exp11_results.json`.
- **Output:**
  - The script creates a directory named `exp11_plots_2`.
  - Inside this directory, it saves a series of `.png` plots, one for each "jump" value tested in the analysis.

### Experiment 12: Polar Stability Boundary Analysis

This experiment investigates the stability boundary of the model by measuring how far the input embedding can be perturbed in different angular directions (in the 2D space spanned by the first two singular vectors) before the model's output changes.

#### `src/exp12_polar_stability_boundary.py`

- **Purpose:** Creates a polar plot showing the "stability boundary" - for each angle θ in the space spanned by the first two singular vectors, it determines the maximum perturbation magnitude that doesn't change the model's hidden state output. This reveals the directional sensitivity of the model to perturbations.
- **Methodology:**
  1. Load Llama model in `float32` precision
  2. Compute Jacobian SVD for the full model (last token embedding → last hidden state)
  3. Extract the first two singular vectors (e₁, e₂) to define a 2D perturbation space
  4. For each angle θ from 0 to 2π (1000 steps):
     - Create perturbation direction: `d = cos(θ)·e₁ + sin(θ)·e₂`
     - Use binary search to find maximum scale `s` where the output doesn't change
     - Refine using ULP-precise (unit in the last place) nextafter for exact float32 boundary
     - Apply perturbation: `x_perturbed = x₀ + s·d`
  5. Generate polar and Cartesian plots showing the stability boundary
- **How to run:**
  ```bash
  python src/exp12_polar_stability_boundary.py
  ```
- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text (hardcoded): "The capital of France is"
  - Number of angles: 1000 (configurable in the script)
  - Maximum search value `s_max`: 1e-6 (configurable)
- **Output Structure:**
  - Creates a timestamped directory in `../results/` (e.g., `../results/exp12_2025-12-18_10-00-00/`)
  - Inside this directory, it saves:
    - **`polar_boundary_data.npz`**: NumPy archive containing:
      - `thetas`: Array of angles (0 to 2π)
      - `max_s_values`: Maximum perturbation magnitude for each angle
      - `low_values`, `high_values`: Binary search bounds
      - `singular_values`: Top 2 singular values
      - `e1`, `e2`: First two singular vectors
      - Metadata: input text, number of angles, parameters
    - **`polar_boundary_data.csv`**: CSV file with columns: theta, max_s, low, high
    - **`polar_stability_boundary.pdf/.png`**: Polar plot of the stability boundary
    - **`stability_boundary_cartesian.pdf/.png`**: Cartesian plot of max_s vs. angle in degrees
  - Prints detailed statistics including mean, median, min, max, and standard deviation of the boundary, along with the angles where min/max occur

#### `src/exp12_part2_radar_plot.py`

- **Purpose:** Loads the data generated by `exp12_polar_stability_boundary.py` and creates multiple visualization plots including radar plots, Cartesian e₁-e₂ space plots, scatter plots, and angle-based plots. This provides different perspectives on the stability boundary shape.
- **How to run:**
  ```bash
  python src/exp12_part2_radar_plot.py <path_to_npz_file>
  ```
  Example:
  ```bash
  python src/exp12_part2_radar_plot.py ../results/exp12_2025-12-18_10-00-00/polar_boundary_data.npz
  ```
- **Inputs:**
  - Path to the `polar_boundary_data.npz` file generated by the main experiment
- **Output Structure:**
  - All plots are saved in the same directory as the input NPZ file:
    - **`radar_plot.pdf/.png`**: Polar radar plot showing the stability boundary with marked cardinal directions (e₁, e₂, -e₁, -e₂) and min/max points
    - **`cartesian_e1_e2_plot.pdf/.png`**: 2D Cartesian plot in e₁-e₂ space with scatter overlay and color-mapped points
    - **`scatter_e1_e2_plot.pdf/.png`**: Dedicated scatter plot with clear axis labels and directional arrows
    - **`degrees_plot.pdf/.png`**: Line plot of max_s vs. angle in degrees (0-360°) with cardinal direction markers
  - Prints detailed statistics to standard output including mean, median, min, max, and values at cardinal directions

#### `src/exp12_part3.py`

- **Purpose:** A simple utility script to load and inspect the polar boundary data. It prints basic statistics and saves the theta-max_s pairs to a CSV file for quick analysis.
- **How to run:**
  ```bash
  python src/exp12_part3.py
  ```
  **Note:** The script has hardcoded paths that need to be edited to point to your specific NPZ file.
- **Inputs:**
  - Path to `polar_boundary_data.npz` (hardcoded in the script)
- **Output:**
  - Prints statistics: non-zero count, min, max, mean, median of max_s values
  - Prints first 10 theta and max_s values for quick inspection
  - Saves a CSV file `polar_boundary_data.csv` with two columns: thetas and max_s

### Experiment 13: Singular Vector Stability Analysis

This experiment analyzes the stability of the model by measuring the maximum perturbation magnitude along each individual singular vector direction (all singular vectors, typically 4096) before the model's output changes. Unlike Experiment 12, which explores a 2D subspace, this experiment comprehensively tests all singular vector directions.

#### `src/exp13_singular_vector_stability.py`

- **Purpose:** For each singular vector direction in the model's Jacobian, determines the maximum perturbation magnitude that doesn't change the model's hidden state output. This provides a complete characterization of the model's directional sensitivity across all principal components of the input-output transformation.
- **Methodology:**
  1. Load Llama model in `float32` precision
  2. Compute Jacobian SVD for the full model (last token embedding → last hidden state)
  3. Extract all singular vectors from Vt (typically 4096 vectors)
  4. For each singular vector eᵢ:
     - Use the singular vector as the perturbation direction
     - Perform binary search to find maximum scale `s` where the output doesn't change
     - Refine using ULP-precise (unit in the last place) nextafter for exact float32 boundary
     - Apply perturbation: `x_perturbed = x₀ + s·eᵢ`
  5. Generate plot showing max_s as a function of singular vector index
- **How to run:**
  ```bash
  python src/exp13_singular_vector_stability.py
  ```
- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text (hardcoded): "The capital of France is"
  - Maximum search value `s_max`: 1e-6 (configurable)
  - Threshold: 0 for exact comparison (configurable)
- **Output Structure:**
  - Creates a timestamped directory in `../results/` (e.g., `../results/exp13_2025-12-18_10-00-00/`)
  - Inside this directory, it saves:
    - **`singular_vector_stability_data.npz`**: NumPy archive containing:
      - `singular_vector_indices`: Array of indices (0 to 4095)
      - `max_s_values`: Maximum perturbation magnitude for each singular vector
      - `low_values`, `high_values`: Binary search bounds for each vector
      - `singular_values_all`: All singular values from the SVD
      - Metadata: input text, s_max, threshold
    - **`singular_vector_stability_data.csv`**: CSV file with columns: singular_vector_index, max_s, low, high
    - **`singular_vector_stability_plot.pdf/.png`**: Line plot showing max_s vs. singular vector index, with mean and median reference lines
  - Prints detailed statistics including:
    - Mean, median, min, max, and standard deviation of max_s across all singular vectors
    - Indices of singular vectors with minimum and maximum stability
    - Top 5 singular values from the SVD

### Experiment 14: Lipschitz Constants via Small Steps Analysis

This experiment investigates the local smoothness and numerical stability of the Llama model by taking extremely small consecutive steps along a principal singular direction and measuring how the output changes. It provides insight into whether the model behaves smoothly at machine precision scales or exhibits discrete jumps due to floating-point rounding errors.

#### `src/exp14_lipschitz_small_steps_analysis.py`

- **Purpose:** Tests whether the model function behaves smoothly at very small scales by taking tiny consecutive steps along the first singular vector direction and measuring consecutive output differences. This reveals the effects of floating-point precision on the model's apparent continuity and Lipschitz behavior.

- **Key Questions:**
  - Is the function locally Lipschitz continuous at floating-point precision?
  - Do consecutive differences scale linearly with step size?
  - Where do numerical precision effects dominate over smooth behavior?
  - Are there discrete jumps in the output despite smooth input changes?

- **Methodology:**
  1. Load Llama model in `float32` precision
  2. Compute Jacobian SVD for the full model (last token embedding → last hidden state)
  3. Extract first right singular vector (direction of maximum sensitivity)
  4. Generate sequence of epsilon values: ε₀, ε₀ + δ, ε₀ + 2δ, ..., ε₀ + Nδ where δ is very small (e.g., 2e-14)
  5. For each epsilon:
     - Perturb input: `x_perturbed = x_original + ε · v₁`
     - Compute output: `y = f(x_perturbed)`
     - Measure consecutive difference: `||y_t - y_{t-1}||`
  6. Analyze patterns:
     - Identify zero-difference steps (no change despite input change)
     - Detect sudden jumps (rounding-induced discontinuities)
     - Compute statistics on difference magnitudes
     - Analyze spacing between jumps

- **How to run:**
  ```bash
  python src/exp14_lipschitz_small_steps_analysis.py \
      --text "The capital of France is" \
      --epsilon-start 1e-6 \
      --epsilon-step 2e-14 \
      --total-steps 500 \
      --use-float64-perturbation True
  ```

  **Command-line arguments:**
  - `--text`: Input text to analyze (default: "The capital of France is")
  - `--epsilon-start`: Starting epsilon value (default: 1e-6)
  - `--epsilon-step`: Step size for epsilon (default: 2e-14)
  - `--total-steps`: Number of steps to take (default: 500)
  - `--use-float64-perturbation`: Use float64 precision for perturbation calculation (default: True)
  - `--output-dir`: Output directory (default: auto-generated with timestamp)

- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Input text: Configurable via `--text` argument
  - Epsilon parameters: Configurable via command-line arguments

- **Output Structure:**
  - Creates a timestamped directory in `../results/` (e.g., `../results/exp14_2025-12-18_10-00-00/`)
  - Inside this directory, it saves:
    - **`lipschitz_small_steps_input_embeddings.npy`**: Input embeddings for all epsilon values
    - **`lipschitz_small_steps_output_embeddings.npy`**: Output embeddings for all epsilon values
    - **`lipschitz_small_steps_difference_norms.npy`**: Consecutive difference norms between outputs
    - **`lipschitz_small_steps_complete.npz`**: Complete data archive containing:
      - `epsilons`: Array of all epsilon values tested
      - `input_embeddings`: All input embeddings
      - `output_embeddings`: All output embeddings
      - `difference_norms`: Consecutive difference norms
      - `singular_values`: Top singular values from the Jacobian SVD
      - `singular_vector_1`: First right singular vector used for perturbations
      - `original_input_embedding`: Original unperturbed embedding
      - Metadata: epsilon_start, epsilon_step, total_steps, use_float64, text
    - **`lipschitz_small_steps_jump_analysis.npz`**: Detailed jump point analysis containing:
      - `jump_indices`: Indices where non-zero differences occur
      - `jump_epsilons`: Epsilon values at jump points
      - `jump_magnitudes`: Magnitudes of differences at jumps
    - **`lipschitz_small_steps_consecutive_differences.pdf`**: Visualization with two subplots:
      - Main plot: Consecutive difference norms vs. steps, with jump points highlighted
      - Zoomed plot: Detailed view around the first jump point
    - **`lipschitz_small_steps_distributions.pdf`**: Distribution analysis with two subplots:
      - Histogram of non-zero difference magnitudes
      - Histogram of spacing between jumps
  - Prints detailed statistics including:
    - Mean, std, min, max of consecutive differences
    - Number and percentage of jump points (non-zero differences)
    - First 10 jump points with epsilon values and difference magnitudes
    - Jump magnitude statistics (mean, std, min, max, median)
    - Jump spacing statistics (mean, std, min, max, median spacing)
    - All jump points with high-precision values

- **Expected Behavior:**
  - **IF** the function were smooth and Lipschitz continuous:
    - Consecutive differences would be approximately constant (proportional to step size)
    - No sudden jumps or zero plateaus
  - **OBSERVED** behavior due to `float32` precision:
    - Most steps have ZERO difference (no change in output)
    - Occasional SPIKES when rounding crosses a threshold
    - Non-uniform spacing of jumps
    - Function appears discontinuous at this scale

- **Use Case:**
  - Understand numerical stability at machine precision scales
  - Quantify how floating-point arithmetic affects LLM behavior
  - Identify critical thresholds where rounding matters
  - Design robust perturbation magnitudes for other experiments

- **Relationship to Other Experiments:**
  - Builds on Experiment 1's SVD framework
  - Complements Experiment 8's perturbation analysis
  - Provides micro-scale view vs Experiment 1's macro comparisons
  - Establishes lower bounds for meaningful perturbation sizes
  - Validates the Lipschitz constant estimates from Experiment 2

### Experiment 15: Stability Boundary L2 Distance Analysis

This experiment extends Experiment 12 by quantifying the magnitude of changes that occur when crossing the stability boundary. It measures the L2 distances between the original and perturbed embeddings, as well as between the original and changed hidden states, at the exact point where the model's output first changes.

#### `src/exp15_stability_boundary_analysis.py`

- **Purpose:** To quantify the magnitude of change in both the input embedding and the final hidden state representation when crossing the stability boundary identified in Experiment 12. This reveals how small the input perturbation is and how large the corresponding output change becomes, providing empirical evidence of the model's numerical instability.

- **Relationship to Experiment 12:**
  - Requires the `.npz` output file from Experiment 12 as input
  - Uses the same angles (thetas) and stability boundary values (max_s) from Experiment 12
  - Analyzes what happens at `min_t = nextafter(max_s)`, the smallest perturbation that causes a change

- **Methodology:**
  1. Load the `.npz` data file generated by `exp12_polar_stability_boundary.py`, which contains:
     - Angles (thetas) in the 2D space spanned by the first two singular vectors
     - Maximum stable perturbation magnitudes (max_s) for each angle
     - Singular vectors e₁ and e₂
  2. Load the Llama model and tokenizer in `float32` precision
  3. Recreate the original input embedding and original final hidden state
  4. For each angle (theta) and its corresponding max_s:
     - Calculate `min_t = nextafter(max_s, inf)`, the next representable float value after max_s
     - This is the smallest perturbation guaranteed to cause a change in the output
     - Construct perturbation direction: `d = cos(θ)·e₁ + sin(θ)·e₂`
     - Apply perturbation: `x_perturbed = x_original + min_t · d`
     - Calculate L2 distance between original and perturbed embeddings: `||x_perturbed - x_original||`
     - Pass perturbed embedding through the model to get the changed final hidden state
     - Calculate L2 distance between original and changed hidden states: `||h_changed - h_original||`
  5. Aggregate and analyze these L2 distances (mean, median, min, max)
  6. Save results and generate visualizations

- **How to run:**
  ```bash
  python src/exp15_stability_boundary_analysis.py <path_to_exp12_npz_file> \
      --precision float64 \
      --output_dir <output_directory>
  ```

  **Examples:**
  ```bash
  # Using float64 precision (recommended)
  python src/exp15_stability_boundary_analysis.py \
      ../results/exp12_2025-12-20_10-55-48/polar_boundary_data.npz \
      --precision float64

  # Using float32 precision
  python src/exp15_stability_boundary_analysis.py \
      ../results/exp12_2025-12-20_10-55-48/polar_boundary_data.npz \
      --precision float32
  ```

  **Command-line arguments:**
  - `npz_file` (required): Path to the `.npz` file from Experiment 12
  - `--output_dir`: Directory to save results (default: auto-generated based on input file and precision)
  - `--precision`: Precision for perturbation calculation, either `float32` or `float64` (default: `float64`)

- **Inputs:**
  - Model path (hardcoded): `/home/chashi/Desktop/Research/My Projects/models/Llama-3.1-8B-Instruct`
  - Experiment 12 NPZ file containing:
    - `thetas`: Array of angles
    - `max_s_values`: Maximum stable perturbation magnitudes
    - `e1`, `e2`: First two singular vectors
    - `input_text`: Original input text

- **Output Structure:**
  - Creates a directory in the same location as the input NPZ file (e.g., `../results/exp15_analysis_polar_boundary_data_float64/`)
  - Inside this directory, it saves:
    - **`boundary_distance_data.npz`**: NumPy archive containing:
      - `thetas`: Array of angles (copied from Experiment 12)
      - `max_s_values`: Maximum stable perturbation magnitudes (copied from Experiment 12)
      - `embedding_l2_distances`: L2 distances between original and perturbed embeddings for each angle
      - `hidden_state_l2_distances`: L2 distances between original and changed hidden states for each angle
      - `input_text`: Original input text
    - **`boundary_distance_data.csv`**: CSV file with columns:
      - `theta`: Angle in radians
      - `max_s`: Maximum stable perturbation magnitude
      - `embedding_l2_dist`: L2 distance in embedding space
      - `hidden_state_l2_dist`: L2 distance in hidden state space
    - **`embedding_l2_distance.png`**: Plot showing embedding L2 distance vs. angle in degrees
    - **`hidden_state_l2_distance.png`**: Plot showing hidden state L2 distance vs. angle in degrees
  - Prints detailed statistics including:
    - Embedding L2 distance statistics: mean, median, min, max
    - Hidden state L2 distance statistics: mean, median, min, max

- **Key Findings (Example Results):**

  Based on the script's documentation, typical results show:

  **Float64 Precision:**
  - Embedding L2 Distance: ~3.86e-12 (mean), range: 2.06e-12 to 8.60e-12
  - Hidden State L2 Distance: ~8.89e-05 (mean), range: 8.31e-05 to 9.89e-05

  **Interpretation:**
  - Input perturbations at the boundary are extremely small (~10⁻¹² scale)
  - Output changes are much larger (~10⁻⁵ scale)
  - This represents an amplification factor of approximately **10⁷** (10 million times)
  - The model exhibits extreme sensitivity: near-imperceptible input changes lead to significant output changes

- **Use Case:**
  - Quantify the amplification of perturbations through the model
  - Understand the relationship between input stability boundaries and output changes
  - Provide empirical evidence for the model's numerical instability
  - Compare stability across different directions in the embedding space
  - Analyze whether certain directions are more or less sensitive than others

- **Relationship to Other Experiments:**
  - **Requires Experiment 12**: Uses the stability boundary data from Experiment 12 as input
  - **Extends Experiment 12**: While Experiment 12 finds where changes occur, Experiment 15 measures how large those changes are
  - **Complements Experiment 2**: Provides concrete L2 distance measurements vs. Experiment 2's Lipschitz constant estimates
  - **Validates Experiment 14**: The large amplification factors support the discontinuous behavior observed in Experiment 14
  - **Relates to Experiment 1**: Provides quantitative measurements of the output differences that Experiment 1 observed qualitatively

### Additional Analysis: PDF Word-by-Word Prediction

- **Purpose:** Analyzes how well a language model predicts each word in a PDF document using preceding context. This helps understand model prediction behavior and the relationship between hidden states and vocabulary embeddings.
- **Script:** [`src/pdf_word_by_word_prediction.py`](./src/pdf_word_by_word_prediction.py)
- **Features:**
  - Extracts text from PDF files using PyMuPDF
  - Performs sequential next-token prediction
  - Context-aware tokenization for proper multi-token word handling
  - Computes logits, probabilities, and cosine similarities
  - Saves results to CSV with progress checkpoints
- **To Run:**
  ```bash
  python src/pdf_word_by_word_prediction.py
  ```
  **Note:** Edit the script to configure:
  - `PDF_PATH`: Path to your PDF file
  - `MODEL_PATH`: Path to your language model
  - `OUTPUT_CSV`: Path for the output CSV file

## Results

The results of the experiments are saved in various formats, including:

- **`.csv`:** Data tables, such as Lipschitz constant measurements.
- **`.npz` and `.npy`:** NumPy arrays for storing embeddings, hidden states, and other numerical data.
- **`.pdf` and `.png`:** Plots and heatmaps visualizing the results.
- **`.json`:** Metadata and structured results.

The naming convention for result files typically includes the experiment number and the parameters used.

## Citation

If you use this code or the results in your research, please cite this repository.

```bibtex
@misc{llm_rounding_error_instability,
  author = {Your Name},
  title = {Analysis of Numerical Instability in Large Language Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/llm_rounding_error_instability}}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
