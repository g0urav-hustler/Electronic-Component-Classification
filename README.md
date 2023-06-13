
<h1  align="center"  > Electronic Component Classification</h1>

  

**WEB URL** https://electronic-component-classification.onrender.com

## Overview

#### It is a project to classify electronics components in the laboratory.

  

----------------------------

## Problem Statement

For the students how they know the component name if it was placed in-front of them and how to search that component.

  

----------------------------

## Goal

To make a Machine Learning model that tells the component with its image.

  

----------------------------

## Technical Aspects

  

#### About the dataset

```

It has different component images into the data folder.

```

  

#### About Model

- There are three different model inceptionv3, mobilenet and xception .

- Best accuracy was given by xception model which is 92%.

  

#### Technology Used

![](https://img.shields.io/badge/Python-3.7-blue.svg)

![](https://img.shields.io/badge/Tensorflow-2.8-blue)

  

----------------------------

## Setup and Intallation

  

Open your terminal and change the directory to project folder

```

$ cd [your-project-folder]

```

Clone the repo in your exiting project folder

```

$ git clone https://github.com/g0urav-hustler/Electronic-Component-Classification.git

```

Making virual environment

```

$ python3 -m [your-virtual-env-name] [project-directory-path]

```

Activate virtual environment

```

$ source [your-virtual-env-name]/bin/activate

```

Install all the requirements

```

$ pip install -r requirements.txt

```

Now your setup has been completed

  
  

----------------------------

## Repository Overview

```

├── LICENSE

├── Makefile <- Makefile with commands like `make data` or `make train`

├── README.md <- The top-level README for developers using this project.

├── data

│ ├── processed <- The final, canonical data sets for modeling.

│ └── raw <- The original, immutable data dump.

│

├── models <- Trained and serialized models, model predictions, or model summaries

│

├── notebooks <- Jupyter notebooks. Naming convention is a number (for ordering),

├── reports <- Generated analysis as HTML, PDF, LaTeX, etc.

│ └── figures <- Generated graphics and figures to be used in reporting

│

├── requirements.txt <- The requirements file for reproducing the analysis environment, e.g.

│

├── setup.py <- makes project pip installable (pip install -e .) so src can be imported

├── src <- Source code for use in this project.

│ ├── __init__.py <- Makes src a Python module

│ │ └── categories.json
│ │ └── read_params.py
│ │ └── make_dataset.py
│ │ └── train_and_evaluate.py
│ │ └── convert_model.py

│

└── tox.ini <- tox file with settings for running tox; see tox.readthedocs.io

```



----------------------------

## License

MIT License

  

Copyright (c) 2021 Gourav Chouhan

  

Permission is hereby granted, free of charge, to any person obtaining a copy

of this software and associated documentation files (the "Software"), to deal

in the Software without restriction, including without limitation the rights

to use, copy, modify, merge, publish, distribute, sublicense, and/or sell

copies of the Software, and to permit persons to whom the Software is

furnished to do so, subject to the following conditions:

  

The above copyright notice and this permission notice shall be included in all

copies or substantial portions of the Software.

  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE

AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

SOFTWARE.

  

----------------------------

### If you like this repo and it is useful, please don't forget to give a ⭐.

