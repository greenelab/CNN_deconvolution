# CNN_deconvolution
Ongoing project on deconvolution using higher-dimensional image representations of gene expression data.

## The following files should run thorugh their respecive bash scripts (run_X.sh):

1. tune.py: This file tunes the models parametrized through the bash script.
1. train.py: This file trains the models parametrized through the bash script.

### Summary
Convolutional neural networks (CNNs)  are highly effective for structured data, leveraging spatial hierarchies to extract meaningful features. However, their strong performance on tasks like genomic sequence modeling and shuffled images suggests they may also work well on unordered data. Multi-layer perceptrons (MLPs), lacking spatial priors, are commonly used for such tasks, but whether CNNs can outperform them when model sizes are controlled remains unclear. This study systematically compares CNNs and MLPs with matched parameter budgets on structured and unstructured datasets, testing their ability to generalize beyond spatially ordered inputs. 

### Research question
Can CNNs outperform MLPs in image classification when both models have a comparable number of trainable parameters? We hypothesize CNNs  will outperform MLPs in intact images with spatial structure, but when we shuffle all columns and rows, MLPs will perform better due to CNNs reliance of spatial structure.

### Environment
Conda environment specs added to /environment folder as yml file. please use:
[conda env create -f env_cnn_.yml]







