# LLM Inference Simulator

This repository contains MATLAB code for simulating and benchmarking distributed inference of Large Language Models (LLMs) across various network and hardware configurations. The simulator supports different block selection, routing, and scheduling algorithms, and provides tools for analyzing throughput, latency, and resource utilization.

## Features

- **Block Placement and Routing Algorithms:**  
  Includes Petals, proposed heuristics algorithm CG-BPRR, and ILP/LP-based methods.
- **Network Topology Simulation:**  
  Supports custom and real-world topologies (see `topology/`).
- **Throughput and Latency Analysis:**  
  Scripts for varying cluster size, request rate, block size, and more.
- **Online and Offline Simulation Modes:**  
  Evaluate both batch and real-time request arrivals.
- **Data and Plotting:**  
  Precomputed results and scripts for generating publication-quality plots.


- **Main MATLAB scripts:** Core simulation and algorithm implementations.
- **`data/`:** Precomputed results and figures.
- **`plot/`:** Plotting scripts and output figures.
- **`topology/`:** Network topology files.

## Getting Started

### Prerequisites

- MATLAB (tested on R2021a and later)
- Optimization Toolbox (gurobi)
- JSON support (`jsondecode`)

### Running a Simulation

1. **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd LLM_inference_simulator
    ```

2. **Edit parameters** in the desired script (e.g., `General_varying_C_online.m`).

3. **Run the script** in MATLAB:
    ```matlab
    General_varying_C_online
    ```

4. **View results** in the MATLAB workspace or as generated plots in the `data/` or `plot/` folders.

### Example Scripts

- `General_varying_C_online.m`: Simulate inference time as a function of cluster size.
- `test_general_case_online.m`: Run a basic online inference benchmark.
- `Petals_online.m`, `WS_RR.m`, etc.: Core algorithm implementations.

## Data Files

- **`throughput_v5.json`**: Hardware throughput profiles.
- **`inter_arrivals.txt`**: Precomputed request arrival times.
- **`data/`**: Contains `.mat` files and figures for analysis.

## Citation

If you use this simulator in your research, please cite the corresponding paper:

@article{sun2026optimizing,
  title={Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models},
  author={Sun, Tingyang and He, Ting and Ji, Bo and Parag, Parimal},
  journal={ACM SIGMETRICS Performance Evaluation Review},
  volume={53},
  number={4},
  pages={12--13},
  year={2026},
  publisher={ACM New York, NY, USA}
}

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or contributions, please open an issue or contact the maintainer.
