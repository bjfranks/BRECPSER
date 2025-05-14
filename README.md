# BRECPSER

## Usage

### Requirements

Tested combination: Python 3.11.5 + PyTorch 2.5.1 + PyTorch_Geometric 2.6.1

Other required Python libraries included: numpy, networkx, loguru, etc.

Assuming a compatible python version you can install the packages as follows:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch_geometric
python -m pip install pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
python -m pip install numpy jupyter notebook networkx loguru tqdm
```

### Dataset Setup

Start by setting up the dataset. This will require the data from [BREC](https://github.com/GraphPKU/BREC/tree/Release/customize/Data/raw) placed into Data/raw. Run:

```bash
python dataset_v4.py
```

Which will create the files brec_3r2r.graphml, brec_CCoHG.graphml, and brec_v3.npy in Data/processed that need to be moved back into Data/raw.

### Evaluate Model

To evaluate a model on some BREC part run 

```bash
python test_BREC.py --part part --name_tag test
```

with part replaced by one of Basic, Regular, Extension, CFI, 4-Vertex_Condition, Distance_Regular, CCoHG, or 3r2r.

Finally, you can use evaluation.ipynb to collect your results and print them.

## Standard Datasets

Use the datasets provided in CCoHG.py and 3r2r.py which are in standard PyTorch Geometric format. The 3r2r dataset has an option to toggle the used target (default diameter) by calling

```python
_3r2r(target="cycles")
```

or if you wish to use the statistics provided as `graph.diameter` and `graph.cycles` yourself

```python
_3r2r(target="None")
```

## Graph Creation

Data/raw/*.graphml already contains the precreated datasets for reproducibility. However, if you wish to recreate or modify them, then the CCoHG_3r2r_generation.ipynb provides all the code to do so.
