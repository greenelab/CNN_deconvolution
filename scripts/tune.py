import os
import json
import ray
import torch
import sys
from ray import tune
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
import argparse
import shutil
import uuid

# Ensure PYTHONPATH includes the parent directory of 'src'
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
print("Updated PYTHONPATH:", sys.path)  # Debugging line

from src.utils import SimpleMLP, SimpleCNN_3CH, SimpleCNN, PCamDataset, load_training_data, get_dimensions

script_dir = Path(__file__).resolve().parent

def train_model(config, 
                model_type, 
                dataset, 
                image_dim, 
                pcam_data_path=None):

    train_loader, val_loader, _, _ = load_training_data(dataset = dataset,
                                                    batch_size=config["batch_size"],
                                                    val_split = 0.2,
                                                    pcam_data_path=pcam_data_path)
    if model_type == "MLP":
        model = SimpleMLP(
            input_dim= image_dim * image_dim,
            fc1_hidden=config["fc1_hidden"],
            fc2_hidden=config["fc2_hidden"],
            fc3_hidden=config["fc3_hidden"]
            )
    elif model_type == 'MLP_3CH':
        model = SimpleMLP(
            input_dim= 3 * image_dim * image_dim,
            fc1_hidden=config["fc1_hidden"],
            fc2_hidden=config["fc2_hidden"],
            fc3_hidden=config["fc3_hidden"]
            )
    elif model_type == 'CNN':
        model = SimpleCNN(
            cha_input=config["cha_input"],
            cha_hidden=config["cha_hidden"],
            fc_hidden=config["fc_hidden"],
            )
    elif model_type == "CNN_3CH":
        model = SimpleCNN_3CH(
            cha_input=config["cha_input"],
            cha_hidden=config["cha_hidden"],
            fc_hidden=config["fc_hidden"]
            )
    else:
        raise ValueError("Invalid model type!")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(25):
        model.train()
        for inputs, labels in train_loader:
            if model_type in ["MLP", "MLP_3CH"]:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs only for MLP
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                if model_type in ["MLP", "MLP_3CH"]:
                    inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs only for MLP
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        accuracy = correct / total
        ray.train.report({"val_loss": val_loss, "accuracy": accuracy})

def run_tuning(model_type, dataset, config, output_path, tmp_dir, working_dir, num_iterations, pcam_data_path=None, image_dim=28):
    
    ray.shutdown()
    os.makedirs(working_dir, exist_ok=True)  # Create the directory if it doesn't exist
    os.makedirs(tmp_dir, exist_ok=True)  # Create the directory if it doesn't exist

    ray.init( 
        runtime_env={
            "working_dir": working_dir,
            "py_modules": [str(script_dir.parent / "src")],
        },
        _temp_dir=tmp_dir,
        log_to_driver=False, include_dashboard=False 
        )

    scheduler = ASHAScheduler(max_t=25, grace_period=5, metric="val_loss", mode="min")

    # Ensure the output dir exists
    output_dir = os.path.dirname(output_path)  # Extract directory from output path
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(num_iterations):

        # Use a unique experiment name for each iteration
        experiment_name = f"tuning_iteration_{i+1}"
        
        # Run tuning
        print(f"Running iteration number {i}...")

        analysis = tune.run(
            tune.with_parameters(train_model, model_type=model_type, dataset=dataset, image_dim=image_dim, pcam_data_path=pcam_data_path),
            config=config,
            num_samples=100,
            scheduler=scheduler,
            resources_per_trial={"cpu": 8},
            fail_fast=False,
            storage_path=tmp_dir,
            name=experiment_name  # Unique experiment name
        )

        # Get the best configuration from the current iteration
        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        print(f"Best hyperparameters for iteration {i+1}: {best_config}")

        # Load existing configurations if the file exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path, "r") as f:
                    existing_configs = json.load(f)
            except (json.JSONDecodeError, IOError):
                print("Warning: Unable to read existing configurations. Starting fresh.")
                existing_configs = []
        else:
            existing_configs = []

        # Append the new configuration
        existing_configs.append(best_config)

        # Save the updated configurations back to the file
        try:
            with open(output_path, "w") as f:
                json.dump(existing_configs, f, indent=4)
            print(f"Configuration for iteration {i+1} saved successfully.")
        except IOError as e:
            print(f"Error saving configuration for iteration {i+1}: {e}")

        # Clean up the temporary directory for storage
        try:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to clean up tmp_dir: {e}")

    ray.shutdown()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for models.")
    parser.add_argument("--model_type", type=str, choices=["MLP", "MLP_3CH", "CNN", "CNN_3CH"], required=True, help="Type of model (MLP, MLP_3CH, CNN).")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "MNISTshuffled", "FashMNIST", "FashMNISTshuffled", "CIFAR10shuffled", "CIFAR10", "PCam", "PCamshuffled"], required=True, help="Dataset to use = NAME + shuffled.")
    parser.add_argument("--tmp_dir", type=str, required=True, help="Temporary directory for Ray Tune.")
    parser.add_argument("--working_dir", type=str, required=True, help="Working directory for Ray Tune. Must have lots of space.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the best configurations.")
    parser.add_argument("--pcam_data_path", type=str, required=True, help="Path to the PCam data.")
    parser.add_argument("--num_iterations", type=str, required=True, help="How many times to tune 100 samples.")

    args = parser.parse_args()

    os.environ["RAY_TMPDIR"] = args.tmp_dir

    config = {
        "MLP": {
            "fc1_hidden": tune.randint(208, 686),  # Number of neurons in the first fully connected layer
            "fc2_hidden": tune.randint(110, 588),  # Number of neurons in the second fully connected layer
            "fc3_hidden": tune.randint(58, 438),  # Number of neurons in the third fully connected layer
            "learning_rate": tune.loguniform(1e-4, 1e-2),  # Learning rate
            "batch_size": tune.choice([32, 64, 120]),  # Batch size
        },
        "MLP_3CH": {
            "fc1_hidden": tune.randint(62, 358),
            "fc2_hidden": tune.randint(40, 218),
            "fc3_hidden": tune.randint(26, 90),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120])
        },
        "CNN": {
            "cha_input": tune.randint(56, 86),  # Number of input channels for conv1
            "cha_hidden": tune.randint(88, 146),  # Number of hidden channels for conv2 and conv3
            "fc_hidden": tune.randint(98, 270),  # Number of hidden units in fully connected layer
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
        },
        "CNN_3CH": {
            "cha_input": tune.randint(40, 80),
            "cha_hidden": tune.randint(64, 128),
            "fc_hidden": tune.randint(128, 256),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120])
        },
        
    }[args.model_type]

    image_dim = get_dimensions(args.dataset)
    num_iterations = int(args.num_iterations)

    run_tuning(model_type=args.model_type, 
                            dataset=args.dataset, 
                            config=config, 
                            output_path= args.output_path, 
                            tmp_dir=args.tmp_dir, 
                            working_dir=args.working_dir,
                            pcam_data_path = args.pcam_data_path, 
                            image_dim=image_dim, num_iterations=num_iterations)