# Compilation

For $\LaTeX$ compilation, it is necessary to generate the _pygments_ for `pymentize` within the `minted` environment.
For this I am using [`sisl/pygments-julia`](https://github.com/sisl/pygments-julia) within a virtual environment. Using
`pipenv` and `gh`, this can be done with:
```sh
$ gh repo clone sisl/pygments-julia
$ pipenv shell
$ pipenv install setuptools
$ python setup.py install
```
Then check that `julia1` is available as a `pygmentize` lexer with
```sh
$ pygmentize -L lexers | rg julia
```
Finally the slides can be compiled with [`tectonic`](https://github.com/tectonic-typesetting/tectonic)
```sh
$ tectonic -X build
```

