# Solving Minimum Spanning Tree Problem in Spiking Neural Networks: Improved Results

This repository contains the official implementation for [Solving Minimum Spanning Tree Problem in Spiking Neural Networks: Improved Results](https://doi.org/10.1109/ICONS62911.2024.00015)

## Reference

If you find our paper or this repository helpful, please consider citing:

```bibtex
@inproceedings{janssenSolvingMinimumSpanning2024,
  author={Janssen, Simon and Groenen, Stijn and Reichert, Simon and Kwisthout, Johan},
  booktitle={2024 International Conference on Neuromorphic Systems (ICONS)},
  title={Solving Minimum Spanning Tree Problem in Spiking Neural Networks: Improved Results},
  year={2024},
  volume={},
  number={},
  pages={47-54},
  keywords={Power demand;Neuromorphics;Neuromorphic engineering;Neurons;Graphics processing units;Spiking neural networks;Parallel processing;Hardware;Complexity theory;Time complexity;Neuromorphic Computing;Graph Algorithms},
  doi={10.1109/ICONS62911.2024.00015}
}

```

## Requirements

To run this project, you need to have Python 3.x installed. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

The spiking neural network algorithm can be run using the following code, where `weight_matrix` is a numpy array containing the weight matrix of the input graph:

```python
from solver import MSTSolver

MSTSolver(weight_matrix).solve()
```

## Testing

The [test.py](test.py) script runs several test cases for the SNN-based minimum spanning tree solver algorithm. It can be run with the following command:

```bash
python test.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
