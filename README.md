# BRECPSER

## Usage

### Requirements

Tested combination: Python 3.11.5 + PyTorch 2.5.1 + PyTorch_Geometric 2.6.1

Other required Python libraries included: numpy, networkx, loguru, etc.

### Dataset Setup

This will require the data from [BREC](https://github.com/GraphPKU/BREC/tree/Release/customize/Data/raw) placed into Data/raw

```bash
python dataset_v4.py
```

### Evaluate Model

```bash
python test_BREC.py
```

## Standard Datasets

Use the datasets provided in CCoHG.py and 3r2r.py which are in standard PyTorch Geometric format. The 3r2r dataset has an option to toggle the used target by calling

```python
_3r2r(target="cycles")
```

or if you wish to use the statistics provided as `graph.diameter` and `graph.cycles` yourself

```python
_3r2r(target="None")
```
