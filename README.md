# Cancer cell line drug response modeling based on multimodal attention networks
Multimodal attention model for modeling drug response based on PPI interactions in cancer cell lines<br/>

# Requriements
pytorch==1.8.0 <br/>
pytorch-geometric==1.6.3 (git master!)<br/>

# Installation
Create a conda virutal env from src folder <br/>
$ conda env create --file ccl_multimodal_attention.yml <br/>
$ conda activate ccl_multimodal_attention <br/>
<br/>
Install Pytorch with <br/>
$ conda install pytorch==1.8.0 cudatoolkit=10.2 -c pytorch <br/>
<br/>
Install Pytorch geometric with corresponding CUDA toolkit version<br/>
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html <br/>
$ pip install git+https://github.com/rusty1s/pytorch_geometric.git <br/>

# Run training
python train_gdsc_benchmark.py --split=blind --seed 42 --batch_size 32 --epochs 300 --gpu 1 <br />
