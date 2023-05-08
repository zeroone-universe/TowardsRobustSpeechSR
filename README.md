# Towards Robust Speech Super-Resolution

This repository contains the unofficial pytorch lightning implementation of the model described in the paper [Towards Robust Speech Super-Resolution](https://web.cse.ohio-state.edu/~wang.77/papers/Wang-Wang.taslp21.pdf) by Heming Wang and Deliang Wang. This implementation includes all three losses proposed in the paper.

## Requirements
 
To run this code, you will need:

- pytorch_lightning==2.0.0
- PyYAML==6.0
- torch==2.0.0
- torchaudio==2.0.0


To automatically install these libraries, run the following command:

```pip install -r requirements.txt```

## Usage

To run the code on your own machine, follow these steps:

1. Open the 'config.yaml' file and modify the file paths, loss type,and hyperparameters as needed.
2. Run the 'main.py' file to start training the model.

The trained model will be saved as ckpt file in 'logger' directory. You can then use the trained model to perform real-time speech frequency bandwidth extension on your own audio wav file by running the 'inference.py' file as

```python inference.py --mode "wav" --path_ckpt <path of checkpoint file> --path_in <path of wav file>```

This repository also support directory-level inference, where the inference is performed on a directory consisting of wav files. You can use the following example to perform directory-level inference,

```python inference.py --mode "dir" --path_ckpt <path of checkpoint file> --path_in <path of directory that contains input wave files> --path_out <path of directory that output files will be saved>```

## Note
- Unfortunately, the performance of my implementation of the model described in the paper is not as satisfactory as expected. If time permits, I will make efforts to improve the model's performance and enhance the implementation if time allows. Any contributions or suggestions are highly appreciated.

## Citation

```bibtex
@ARTICLE{9335252,
  author={Wang, Heming and Wang, DeLiang},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Towards Robust Speech Super-Resolution}, 
  year={2021},
  volume={29},
  number={},
  pages={2058-2066},
  doi={10.1109/TASLP.2021.3054302}}
```
