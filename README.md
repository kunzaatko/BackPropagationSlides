# Deep Learning Introduction

This repository contains a LaTeX presentation on the fundamentals of neural networks, including notation, backpropagation algorithm, stochastic gradient descent, and practical examples.

## Examples Covered
- Binary classifier on 2D feature space
- Convolutional neural network on MNIST dataset

## Figures
- Mathematical activation functions
![`src/figs/math_activations.pdf`](src/figs/math_activations.pdf)
- Gate activation functions
![`src/figs/gate_activations.pdf`](src/figs/gate_activations.pdf)
- Variants of ReLU activation
![`src/figs/ReLU_variants.pdf`](src/figs/ReLU_variants.pdf)
- 2D classification dataset
![`src/figs/2d_classification_data.pdf`](src/figs/2d_classification_data.pdf)
- Training animation video
![`src/video/training_animation.mp4`](src/video/training_animation.mp4)

## Compilation

For $\LaTeX$ compilation, it is necessary to generate the Julia _lexer_ for `pymentize` for the `minted` environment.
For this I am using [`sisl/pygments-julia`](https://github.com/sisl/pygments-julia) within a virtual environment. Using
`pipenv` and `gh`, this can be done with:
```sh
$ gh repo clone sisl/pygments-julia
$ pipenv shell
$ cd pygments-julia
$ pipenv install setuptools
$ python setup.py install
```
Then check that `julia1` is available as a `pygmentize` lexer with
```sh
$ pygmentize -L lexers | rg julia1
```

You must ensure that the directory for running the `pygmentize` external command exists. This can be set to run in
the base directory in `Tectonic.toml` or you must create the directory `shell-escape` manually with
```sh
$ mkdir shell-escape
```

Finally the slides can be compiled with [`tectonic`](https://github.com/tectonic-typesetting/tectonic)
```sh
$ tectonic -X build
```

Keep in mind that the virtual environment must be active for the compilation to work.
