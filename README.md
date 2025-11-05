# Deep Learning Introduction

This repository contains a LaTeX presentation on the fundamentals of neural networks, including notation, backpropagation algorithm, stochastic gradient descent, and practical examples.

The PDF with the latest version of the slides can be found [here](https://github.com/kunzaatko/BackPropagationSlides/releases/latest/download/DL_intro_slides.pdf).

## Examples Covered
- Binary classifier on 2D feature space
- Convolutional neural network on MNIST dataset

## Figures
<p align="center">
<a href="src/figs/math_activations.pdf">
<img alt="math_activations" src="https://github.com/user-attachments/assets/8c61d64a-5b90-4590-b6d2-3fea7d724e17" width="30%" />
</a>
&nbsp; &nbsp;
<a href="src/figs/gate_activations.pdf">
<img alt="gate_activations" src="https://github.com/user-attachments/assets/f5ccb534-e03c-494a-a5fd-b1f64bd5e5d3" width="30%" />
</a>
&nbsp; &nbsp;
<a href="src/figs/ReLU_variants.pdf">
<img alt="ReLU_variants" src="https://github.com/user-attachments/assets/d5df8419-ad2f-4503-900f-ff6cac4bce8c" width="30%" />
</a>
</p>

<p align="center">
<a href="src/figs/2d_classification_data.pdf">
<img alt="2d_classification_data" src="https://github.com/user-attachments/assets/cced890a-003d-4cdc-bd63-8f1d7c5f3a6f" width="45%" />
</a>
</p>

- [Training animation video](src/video/training_animation.mp4)

https://github.com/user-attachments/assets/32734f01-1f5d-4602-8cce-7757a7198edf

## Compilation
For $\LaTeX$ compilation, it is necessary to generate the Julia _lexer_ for `pymentize` for the `minted` environment.
For this I am using [`sisl/pygments-julia`](https://github.com/sisl/pygments-julia) within a virtual environment.
Using `uv`, you can install the necessary `Pygments` lexer by running
```sh
uv pip install --system git+https://github.com/sisl/pygments-julia#egg=pygments_julia 
```
Then check that `julia1` is available as a `pygmentize` lexer with
```sh
pygmentize -L lexers | rg julia1
```

> [!NOTE]
> You must also have the `Pygments` python package installed. 
> Follow the instructions in the [documentation](https://pygments.org/) to install the package.

You must ensure that the directory for running the `pygmentize` external command exists. This can be set to run in
the base directory in `Tectonic.toml` or you must create the directory `shell-escape` manually with
```sh
mkdir shell-escape
```

Finally the slides can be compiled with [`tectonic`](https://github.com/tectonic-typesetting/tectonic)
```sh
tectonic -X build
```

> [!NOTE]
> You can also install the necessary python packages within a virtual environment. You can use `uv` for this by running
> `uv venv` and then installing the packages without the `--system` flag. Keep in mind that the virtual environment must
> be active for the compilation to work.
