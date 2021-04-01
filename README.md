# Cancer cell line drug response modeling based on multimodal attention networks
Multimodal attention model for modeling PPI interaction in cancer cell lines<br/>

# Requriements
pytorch==1.8.0 <br/>
pytorch-geometric==1.6.3 <br/>

# Installation
Create a conda or virutal env <br/>
$ conda create -n cancer_multimodal_attention <br/>
$ conda activate cancer_multimodal_attention <br/>
<br/>
Install Pytorch with <br/>
$ conda install pytorch cudatoolkit=10.2 -c pytorch <br/>
<br/>
Install Pytorch geometric with corresponding CUDA version<br/>
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-geometric==1.6.3 <br/>
<br/>
Install required depenencies <br/>
$ conda env update --file CCL_multimodal_attention.yml <br/>
