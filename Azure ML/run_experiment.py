# run_experiment.py

from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import DockerConfiguration

def main():
    """
    Main function to set up and run an Azure Machine Learning experiment.
    """
    # Connect to Azure ML workspace using the configuration file
    ws = Workspace.from_config()
    print('Connected to Azure ML workspace.')

    # Create a new experiment in the workspace
    experiment_name = 'supply-chain-optimization'
    experiment = Experiment(workspace=ws, name=experiment_name)
    print(f'Experiment "{experiment_name}" created.')

    # Set up the training environment using a Conda environment specification
    env = Environment.from_conda_specification(name='supply_chain_env', file_path='environment.yml')
    print('Training environment set up.')

    # Define the Docker configuration (optional, uncomment if needed)
    # docker_config = DockerConfiguration(use_docker=True)

    # Set up the ScriptRunConfig for the experiment
    script_config = ScriptRunConfig(
        source_directory='.',  # Directory where the training script is located
        script='train.py',     # Training script to run
        arguments=['--data-path', dataset.as_named_input('input_data').as_mount()],  # Arguments to pass to the training script
        environment=env,       # Environment to use for running the script
        compute_target='your-compute-target'  # Compute target (e.g., an Azure ML compute cluster)
        # docker_runtime_config=docker_config  # Docker configuration (optional)
    )
    print('ScriptRunConfig set up.')

    # Submit the experiment and wait for completion
    run = experiment.submit(config=script_config)
    print('Experiment submitted.')
    run.wait_for_completion(show_output=True)
    print('Experiment completed.')

    # Get and print the metrics from the run
    metrics = run.get_metrics()
    print('Run metrics:', metrics)

    # Register the model from the run output
    model_name = 'supply_chain_rf_model'
    model_path = 'outputs/rf_model.joblib'
    model = run.register_model(model_name=model_name, model_path=model_path)
    print(f'Model registered: {model.name}, ID: {model.id}, Version: {model.version}')

if __name__ == '__main__':
    main()
