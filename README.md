## Installation
Providing requirements.txt is a tricky task since some libraries require dependencies with different versions.
Because of this, it is recommended to install all libraries and dependencies on demand.

## Structure overview
[**AM.py**](https://github.com/seelennebel/llm/blob/main/AM.py) - Python script to train the Anonymization Model  

[**application.ipynb**](https://github.com/seelennebel/llm/blob/main/application.ipynb) - Jupyter Notebook containing GUI for outputting AM  

[**evaluation.ipynb**](https://github.com/seelennebel/llm/blob/main/evaluation.ipynb) - Jupyter Notebook containing an evaluation of the trained model  

[**labels.py**](https://github.com/seelennebel/llm/blob/main/labels.py) - Python file that contains constant id2labels and labels2id dictionaries. !!! DO NOT CHANGE THIS FILE !!!  

[**model_checkpoint.py**](https://github.com/seelennebel/llm/blob/main/model_checkpoint.py) - Python file that contains a specific pre-trained DistilBERT checkpoint. !!! DO NOT CHANGE THIS FILE !!!

[**output.ipynb**](https://github.com/seelennebel/llm/blob/main/output.ipynb) - Jupyter Notebook containing a function that is used in **application.ipynb** and used for outputting AM  

[**preprocessing.ipynb**](https://github.com/seelennebel/llm/blob/main/preprocessing.ipynb) - Jupyter Notebook that can be launched to create the dataset that the model would be trained on  

**push_AM_to_hub.py and push_tokenizer_to_hub** - Python files that have pure utility functions
