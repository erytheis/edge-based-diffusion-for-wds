# Steady State Estimation in Water Distribution Systems with Edge-based Diffusion




## Roadmap
- [x] Diffusion Model
- [ ] Requirements
- [ ] Pressure reconstruction
- [ ] Generated Data

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/erytheis/edge-based-diffusion-for-wds.git
   cd edge-based-diffusion-for-wds
   ```

2. **Set up a Python environment** (conda or venv recommended):
   ```bash
   conda create -n erytheis_env python=3.10
   conda activate erytheis_env
   ```

3. **Install required packages**:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install -r requirements.txt
   ```
   (Make sure to include `requirements.txt` or similar in the repo with PyTorch, NumPy, SciPy, and other dependencies.)


## Usage

1. **Running a Basic Simulation**  
   Use `main.py` to run a simple, demonstrative edge-based diffusion simulation:
   ```bash
   python main.py --config config/your_experiment.yaml
   ```
   This script will:
   - Parse the configuration file at input/config.yaml
   - Apply the edge-based diffusion solver on a GPU (if available).  
   - Calculate $R^2$ with values obtained from WNTR (if available)