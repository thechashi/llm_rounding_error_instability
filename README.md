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

## Experiments

### Experiment 1: Layer-wise SVD Perturbation Analysis

*   **Purpose:** Analyzes how perturbations propagate through the layers of the model using SVD analysis.
*   **Script:** [`src/exp1_layerwise_svd_perturbation_analysis.py`](./src/exp1_layerwise_svd_perturbation_analysis.py)
*   **To Run:**
    ```bash
    python src/exp1_layerwise_svd_perturbation_analysis.py
    ```

### Experiment 2: Average Lipschitz Constant Computation

*   **Purpose:** Estimates the local Lipschitz constant to quantify the model's sensitivity to input changes.
*   **Scripts:**
    *   [`src/exp2_average_lipschitz_constant_computation.py`](./src/exp2_average_lipschitz_constant_computation.py)
    *   [`src/exp2_v2_average_lipschitz_constant_computation.py`](./src/exp2_v2_average_lipschitz_constant_computation.py) (Alternative version)
*   **To Run:**
    ```bash
    python src/exp2_average_lipschitz_constant_computation.py
    ```

### Experiment 3: FP32 Precision Evaluation

*   **Purpose:** Investigates the effect of floating-point precision on perturbations and model behavior.
*   **Script:** [`src/exp3_fp32_precision_evaluation.py`](./src/exp3_fp32_precision_evaluation.py)
*   **To Run:**
    ```bash
    python src/exp3_fp32_precision_evaluation.py
    ```

### Experiment 4: GPU Output Comparison and Divergence Analysis

*   **Purpose:** Compares the outputs of the same model running on different GPUs to check for hardware-specific instabilities and analyzes where divergence occurs.
*   **Scripts:**
    *   **Part 1:** [`src/exp4_part1_gpu_output_generation.py`](./src/exp4_part1_gpu_output_generation.py) - Generates model outputs on different GPUs
    *   **Part 2:** [`src/exp4_part2_gpu_output_comparison.py`](./src/exp4_part2_gpu_output_comparison.py) - Compares outputs across GPUs
    *   **Part 3:** [`src/exp4_part3_model_output_divergence_analysis.py`](./src/exp4_part3_model_output_divergence_analysis.py) - Analyzes divergence points
    *   **Part 4:** [`src/exp4_part4_internal_representation_plotting.py`](./src/exp4_part4_internal_representation_plotting.py) - Plots internal representations
    *   **Part 5:** [`src/exp4_part5_embedding_perturbation_until_divergence.py`](./src/exp4_part5_embedding_perturbation_until_divergence.py) - Tests perturbations until divergence
    *   **Part 6:** [`src/exp4_part6_embedding_similarity_comparison.py`](./src/exp4_part6_embedding_similarity_comparison.py) - Compares embedding similarities
*   **Dependencies:**
    *   Part 2-6 depend on outputs from previous parts in the sequence
    *   Part 1 generates GPU-specific output files
    *   Part 2 compares outputs and identifies divergence indices
    *   Parts 3-6 use the divergence analysis results from previous parts
*   **To Run:** The scripts in this experiment **must** be run in sequence:
    ```bash
    # Step 1: Generate outputs on different GPUs
    python src/exp4_part1_gpu_output_generation.py

    # Step 2: Compare GPU outputs
    python src/exp4_part2_gpu_output_comparison.py

    # Step 3: Analyze where outputs diverge
    python src/exp4_part3_model_output_divergence_analysis.py

    # Step 4: Plot internal representations
    python src/exp4_part4_internal_representation_plotting.py

    # Step 5: Test embedding perturbations
    python src/exp4_part5_embedding_perturbation_until_divergence.py

    # Step 6: Compare embedding similarities
    python src/exp4_part6_embedding_similarity_comparison.py
    ```

### Experiment 5: Layer-wise and Module-wise Lipschitz Analysis

*   **Purpose:** Drills down into specific layers and submodules (self-attention, MLP) to compute Lipschitz constants and see where perturbations have the most impact.
*   **Script:** [`src/exp5_layerwise_modulewise_lipschitz_analysis.py`](./src/exp5_layerwise_modulewise_lipschitz_analysis.py)
*   **To Run:**
    ```bash
    python src/exp5_layerwise_modulewise_lipschitz_analysis.py
    ```

### Experiment 6: Optimization for Equal Logits and Token Flip Analysis

*   **Purpose:** Finds an input embedding that results in equal logits/probabilities for the top two predicted tokens, then analyzes what causes token prediction flips through Jacobian SVD analysis.
*   **Scripts:**
    *   **Part 1a:** [`src/exp6_part1_optimization_for_equal_probabilities.py`](./src/exp6_part1_optimization_for_equal_probabilities.py) - Direct logit optimization approach
    *   **Part 1b:** [`src/exp6_part1_geometric_approach_for_equal_logits.py`](./src/exp6_part1_geometric_approach_for_equal_logits.py) - Geometric orthogonality constraint approach
    *   **Part 2:** [`src/exp6_part2_token_flip_causation_analysis.py`](./src/exp6_part2_token_flip_causation_analysis.py) - Analyzes token flips via Jacobian SVD
*   **Dependencies:**
    *   Part 2 requires the optimized embeddings file (`optimized_state_geometric.pt`) generated by Part 1b
*   **To Run:**
    ```bash
    # Step 1: Choose either approach for Part 1:
    python src/exp6_part1_optimization_for_equal_probabilities.py
    # OR (recommended for Part 2)
    python src/exp6_part1_geometric_approach_for_equal_logits.py

    # Step 2: Update the file path in Part 2 script to point to the generated optimized_state file
    # Then run Part 2 using the optimized embeddings:
    python src/exp6_part2_token_flip_causation_analysis.py
    ```

### Experiment 7: Logit Difference Heatmaps and Randomness Quantification

*   **Purpose:** Creates 2D heatmaps of the logit difference landscape around a point of instability, zooms into regions of interest, and quantifies the randomness and chaos in the decision boundary.
*   **Scripts:**
    *   **Part 1:** [`src/exp7_part1_logit_difference_heatmap_generation.py`](./src/exp7_part1_logit_difference_heatmap_generation.py) - Generates full heatmaps
    *   **Part 2:** [`src/exp7_part2_zoomed_in_logit_heatmap_plot.py`](./src/exp7_part2_zoomed_in_logit_heatmap_plot.py) - Zooms into specific regions
    *   **Part 3:** [`src/exp7_part3_output_randomness_quantification.py`](./src/exp7_part3_output_randomness_quantification.py) - Quantifies randomness metrics
*   **Dependencies:**
    *   Part 2 and Part 3 both require the `.npz` matrix files generated by Part 1 (e.g., `logit_diff_1st_2nd.npz`, `logit_diff_1st_10th.npz`, `logit_diff_1st_4096th.npz`)
*   **To Run:**
    ```bash
    # Step 1: Generate heatmaps and save matrices
    python src/exp7_part1_logit_difference_heatmap_generation.py
    # This will create .npz files in the results directory

    # Step 2 (Optional): Zoom into regions of interest
    python src/exp7_part2_zoomed_in_logit_heatmap_plot.py <npz_file> --e1_start <val> --e1_end <val> --e2_start <val> --e2_end <val>
    # Example:
    # python src/exp7_part2_zoomed_in_logit_heatmap_plot.py results/exp7_<timestamp>/logit_diff_1st_2nd.npz --e1_start -5e-7 --e1_end 5e-7 --e2_start -5e-7 --e2_end 5e-7

    # Step 3: Quantify randomness in the decision boundary
    python src/exp7_part3_output_randomness_quantification.py <npz_file>
    # Example:
    # python src/exp7_part3_output_randomness_quantification.py results/exp7_<timestamp>/logit_diff_1st_2nd.npz
    ```

### Additional Analysis: PDF Word-by-Word Prediction

*   **Purpose:** Analyzes how well a language model predicts each word in a PDF document using preceding context. This helps understand model prediction behavior and the relationship between hidden states and vocabulary embeddings.
*   **Script:** [`src/pdf_word_by_word_prediction.py`](./src/pdf_word_by_word_prediction.py)
*   **Features:**
    *   Extracts text from PDF files using PyMuPDF
    *   Performs sequential next-token prediction
    *   Context-aware tokenization for proper multi-token word handling
    *   Computes logits, probabilities, and cosine similarities
    *   Saves results to CSV with progress checkpoints
*   **To Run:**
    ```bash
    python src/pdf_word_by_word_prediction.py
    ```
    **Note:** Edit the script to configure:
    *   `PDF_PATH`: Path to your PDF file
    *   `MODEL_PATH`: Path to your language model
    *   `OUTPUT_CSV`: Path for the output CSV file

## Results

The results of the experiments are saved in various formats, including:

*   **`.csv`:** Data tables, such as Lipschitz constant measurements.
*   **`.npz` and `.npy`:** NumPy arrays for storing embeddings, hidden states, and other numerical data.
*   **`.pdf` and `.png`:** Plots and heatmaps visualizing the results.
*   **`.json`:** Metadata and structured results.

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