<h3 align="center">Co-author prediction</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()

</div>

---

<p align="center"> 
A graph neural network which uses a neo4j database for link prediction
    <br> 
</p>

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About ](#-about-)
- [🏁 Getting Started ](#-getting-started-)
  - [Prerequisites](#prerequisites)
- [🔧 Running the Code ](#-running-the-code-)
- [🎈 Usage ](#-usage-)
- [🚀 Deployment ](#-deployment-)
- [⛏️ Built Using ](#️-built-using-)
- [✍️ Authors ](#️-authors-)

## 🧐 About <a name = "about"></a>

A graph neural network which predicts upto n number of co-authors, one particular author would want to work with. This model has been leveraged in the form of a notebook, python script and a flask api, all of which will be explained in their respective sections ahead.

## 🏁 Getting Started <a name = "getting_started"></a>

### Prerequisites

Before cloning the project, I highly recommend using a virtual environment, since the scale of this project was not very large, I have used `pip`.
If this project has a larger scale or is to be worked on by others, I would recommned `conda` instead.

First create a python env and install all the dependencies
```bash
python3 -m venv venv
# activate env
source venv/bin/activate
pip install -r requirements.txt
```

## 🔧 Running the Code <a name = "tests"></a>

Once the dependencies have been installed, I highly recommend using JupyterLab for running the notebooks. The purpose of each notebook will be explained in the section below.

## 🎈 Usage <a name="usage"></a>

<font size=3>**All the information about the code has been included in the notebooks, some commens have also been included in places where I seemed necessary**</font>

- `notebooks/data_visualisation.ipynb` : consists of all the relevant to visualise data and find outliers using a visual means
- `notebooks/dgl_data_creation.ipynb`: This notebook takes the data in the csv format and converts it to a DGL data which can loaded into the dataloader directly
- `notebooks/inference.ipynb`:The code used for inference of the model, the saved model is loaded and then used
- `notebooks/model_dgl_data.ipynb`: Taking in the dgl data and creating a GraphSage model, training the neural network and saving it for inference
- `notebooks/model_training.ipynb`: An alternate script to train the csv data using pytorch geometric
- `notebooks/tabular_data_creation.ipynb`:Connects to the local neo4j database and converts to useful csv data

<hr>

`scripts/inference.py`: A python inference script which can be imported as a module into other scripts
`scripts/model_definition.py`: The model definition from which the GraphSage model can be imported during deployment
`api/main.py`: Consists of the flask api for this model.
It returns 5(can be chanegd) author ids along with a score 

## 🚀 Deployment <a name = "deployment"></a>

A flask api which can be deployed extremely easily

Directions to run the api locally
```bash
cd api
python main.py
```

Example api call:
```
http://localhost:5000/get_possible_coauthors?author_id=authorID_5ef6f_df325_13aa7_cd11f_72bec
```



## ⛏️ Built Using <a name = "built_using"></a>

- [Python3](https://www.python.org/) - Language
- [Pytorch](https://pytorch.org/) - Deep Learning framework
- [DGL](https://github.com/dmlc/dgl) - Graph Neural Networks Ecosystem
- [Flask](https://flask.palletsprojects.com/en/2.3.x/) - Api creation

## ✍️ Authors <a name = "authors"></a>

- [@viraj-s15](https://github.com/viraj-s15/) 



