import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import sys
import os
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
from src import SoftOrdering1DCNN, RNASeqData

def get_dimensions(dataset):
    ''' Get image dimensions from dataset name'''
    if "MNIST" in dataset:
        image_dim = 28
    elif "PCam" in dataset:
        image_dim = 96
    elif "CIFAR10" in dataset:
        image_dim = 32
    else:
        raise ValueError('Data must be MNIST, FashMNIST, PCam or CIFAR10')
    return image_dim

def make_pseudobulks(adata, number_of_bulks, num_cells, prop_type, noise):
    """
    Create pseudobulks from single-cell RNA-seq data.

    Parameters
    ----------
    adata (anndata.AnnData): Single-cell RNA-seq data.
    number_of_bulks (int): Number of pseudobulks to create.
    num_cells (int): Number of cells to use for each pseudobulk.
    prop_type (str): Type of proportions ('random' or 'real').
    noise (bool): Whether to add Gaussian noise to the pseudobulks.

    Returns
    -------
    pandas.DataFrame: DataFrame of pseudobulks.
    pandas.DataFrame: DataFrame of cell type proportions for each pseudobulk.

    """
    cell_types = adata.obs["cell_types"].unique()
    gene_ids = adata.var.index
    n = len(cell_types)
    pseudobulks = []
    proportions = []

    for num in range(number_of_bulks):
        if prop_type == "random":
            prop_vector = np.random.dirichlet(np.ones(n))
            prop_vector = np.round(prop_vector, decimals=5)
            cell_counts = (prop_vector * num_cells).astype(int)
            while np.any(cell_counts == 0):
                prop_vector = np.random.dirichlet(np.ones(n))
                prop_vector = np.round(prop_vector, decimals=5)
                cell_counts = (prop_vector * num_cells).astype(int)
        elif prop_type == "realistic":
            cell_type_counts = adata.obs["cell_types"].value_counts(normalize=True)
            noise_ = np.random.normal(0, 0.01, cell_type_counts.shape)
            prop_vector = cell_type_counts[cell_types].values + noise_

            # Ensure no negative proportions and normalize
            prop_vector = np.maximum(prop_vector, 0)
            prop_vector = prop_vector / prop_vector.sum()

            cell_counts = (prop_vector * num_cells).astype(int)
            while np.any(cell_counts == 0):
                prop_vector = np.random.dirichlet(np.ones(n))
                prop_vector = prop_vector / prop_vector.sum()
                cell_counts = (prop_vector * num_cells).astype(int)
        elif prop_type == "single_cell":
            # Set a high proportion for one cell type and distribute the remainder
            prop_vector = np.zeros(n)
            dominant_cell_type = np.random.choice(n)
            dominant_proportion = np.random.uniform(0.7, 0.95)  # Randomly choose a high proportion
            prop_vector[dominant_cell_type] = dominant_proportion
            remaining_prop = 1 - dominant_proportion  # Distribute the remaining proportion

            # Distribute remaining proportion across other cell types
            other_indices = [i for i in range(n) if i != dominant_cell_type]
            prop_vector[other_indices] = np.random.dirichlet(np.ones(len(other_indices))) * remaining_prop
            prop_vector = np.round(prop_vector, decimals=5)
            cell_counts = (prop_vector * num_cells).astype(int)
        elif prop_type == "zeros":
            prop_vector = np.ones(n)
            #  Randomly choose cells to set to 0 (at least 1 and at most n-2)
            num_zero_cells = np.random.randint(1, 2)
            zero_indices = np.random.choice(range(n), size=num_zero_cells, replace=False)
            # Set chosen indices to 0
            prop_vector[zero_indices] = 0
            # Senerate random proportions for non-zero cells
            non_zero_indices = [i for i in range(n) if i not in zero_indices]
            prop_vector[non_zero_indices] = np.random.dirichlet(np.ones(len(non_zero_indices)))
            prop_vector = np.round(prop_vector, decimals=5)
            cell_counts = (prop_vector * num_cells).astype(int)
        else:
            raise ValueError("prop_type must be either 'random' 'realistic' 'single_cell' 'zeros")

        sampled_cells = []
        for cell_type, count in zip(cell_types, cell_counts):
            sampled_cells.append(adata[adata.obs["cell_types"] == cell_type].X[
                    np.random.choice(
                        adata[adata.obs["cell_types"] == cell_type].shape[0],
                        count,
                        replace=len(adata[adata.obs["cell_types"] == cell_type]) < count,),
                    :,].toarray())

        pseudobulk = np.sum(np.vstack(sampled_cells), axis=0).astype(float)
        if noise:
            pseudobulk += np.random.normal(0, 0.05, pseudobulk.shape)
            pseudobulk = np.clip(pseudobulk, 0, None)  # Ensure non-negative values

        pseudobulks.append(pseudobulk)
        proportions.append(prop_vector)

    pseudobulks_df = pd.DataFrame(pseudobulks, columns=gene_ids)
    proportions_df = pd.DataFrame(proportions, columns=cell_types)

    return pseudobulks_df, proportions_df


def train_model(config, train_expressions, train_proportions, test_expressions, test_proportions):
    """
    Trains a model using the given hyperparameter configuration and data.

    Args:
        config (dict): A dictionary containing the hyperparameter configuration.
        train_expressions (list): A list of training expressions.
        train_proportions (list): A list of training proportions.
        test_expressions (list): A list of test expressions.
        test_proportions (list): A list of test proportions.

    Returns:
        None
    """

    # Initialize the data module with the current hyperparameter configuration
    data_module = RNASeqData(
        train_expressions=train_expressions,
        train_proportions=train_proportions,
        test_expressions=test_expressions,
        test_proportions=test_proportions,
        batch_size=config["batch_size"],
        n_splits=config["n_splits"]
    )

    # Initialize the model with the current hyperparameter configuration
    model = SoftOrdering1DCNN(
        input_dim=data_module.input_dim,
        output_dim=data_module.output_dim,
        sign_size=config["sign_size"],
        cha_input=config["cha_input"],
        cha_hidden=config["cha_hidden"],
        K=config["K"],
        dropout_input=config["dropout_input"],
        dropout_hidden=config["dropout_hidden"],
        dropout_output=config["dropout_output"],
        learning_rate=config["learning_rate"]
    )

    trainer = pl.Trainer(
        max_epochs=25,
        callbacks=[TuneReportCallback({"val_loss": "val_loss"})]
    )

    # Run training with the data module
    trainer.fit(model, datamodule=data_module)