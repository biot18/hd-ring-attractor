# HD-Ring-Attractor

An open-source Python implementation of the head-direction ring attractor model  
based on “Internally Organized Mechanisms of the Head Direction Sense” (Peyrache et al., 2015).

## Features
- Continuous attractor network simulation  
- Von Mises tuning curve fitting  
- Bayesian decoding of head direction  
- Visualization and animation of activity packets  

## Installation

```bash
# clone
git clone https://github.com/biot18/hd-ring-attractor
cd hd-ring-attractor

# conda
conda env create -f environment.yml
conda activate hd-attractor

# pip
pip install -r requirements.txt