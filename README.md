# CCL-multimodal-attention
Multimodal attention model for modeling PPI interaction in cancer cell lines

# Requriements
pytorch=1.8.0 
pytorch-geometric=1.6.3

# Installation
Create a conda or virutal env
$ conda create -n cancer_multimodal_attention
$ conda activate cancer_multimodal_attention

Install Pytorch with 
$ conda install pytorch cudatoolkit=10.2 -c pytorch

Install Pytorch geometric with corresponding CUDA version
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
$ pip install torch-geometric

Install required depenencies
$ conda env update --file CCL_multimodal_attention.yml
