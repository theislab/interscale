import wandb
import pandas as pd

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def load_result_as_df(sweep_id, sweep_goal: str):
    """
    sweep_id: str - ID from WandB run
    sweep_goal: robustenss, parameter
    """
    api = wandb.Api()
    entity, project = "francesca-drummer", "InterScale_hyperparameter_sweep"  

    # Get all runs associated with the sweep
    sweep_runs = api.sweep(f"{entity}/{project}/{sweep_id}").runs

    data = []
    for run in sweep_runs:
        if run.state == 'finished':
            prediction_task = run.config['cfg']['dataset']['prediction_task']
        
            run_data = {
                "id": run.id,
                "name": run.name,
                "seed": run.config.get("model.optim.seed", None),
                "state": run.state,  # finished, running, failed
                "pct_mask_nodes": run.config.get("dataset.pct_mask_nodes", None),
                "radius": run.config.get("dataset.spatial_neigbors_kwargs.radius", None),
                "decoder_type": run.config.get("model.decoder.type", None),
                "runtime_seconds": run.summary.get("_runtime", None),
                "total_parameters": run.summary.get("total_parameters", None),
            }
            if 'regression' in prediction_task:
                run_data.update({
                    "test_r2": run.summary.get("test_r2", None),
                    "test_pearson_corr": run.summary.get("test_pearson_corr", None),
                })
            elif 'classification' in prediction_task:
                num_classes = run.config['cfg']['dataset']['num_classes']
                run_data.update({
                    "test_acc": run.summary.get("test_acc", None),
                    "test_f1_micro/avg": run.summary.get("test_f1_micro/avg", None)
                })
                for class_idx in range(num_classes):
                    run_data[f'test_f1/class_{class_idx}'] = run.summary.get(f"test_f1/class_{class_idx}")
    
            if 'graph' in prediction_task:
                run_data.update({
                    "split_key": run.config.get("dataset.split_key", None)
                })
            if sweep_goal == 'parameter':
                if 'gnn' in run_data['name']:
                    run_data.update({
                        "gnn_num_layers": run.config.get("gnn.num_layers", None),
                        "gnn_hidden_dim": run.config.get("gnn.hidden_dim", None),
                        "embed_dim": run.config.get("gnn.embed_dim", None),
                    })
                if 'transformer' in run_data['name']:
                    run_data.update({
                        "trans_n_heads": run.config.get("transformer.n_heads", None),
                        "trans_num_layers": run.config.get("transformer.num_layers", None),
                        "trans_dim_feedforward": run.config.get("transformer.dim_feedforward", None),
                    })
            
            data.append(run_data)
        
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def compute_mean_and_std(df):
    # Compute mean and standard deviation
    df_agg = df.groupby(["pct_mask_nodes", "radius", "decoder_type"]).agg(
        mean_test_r2=("test_r2", "mean"),
        std_test_r2=("test_r2", "std"),
        mean_test_pearson=("test_pearson_corr", "mean"),
        std_test_pearson=("test_pearson_corr", "std"),
        mean_run_time=("runtime_seconds", "mean"),
        std_run_time=("runtime_seconds", "std"),
    ).reset_index()

    return df_agg

def summary_df(df, metric, decoder_type = "linear"):
    """
    metric: str - column in df
    """
    # Filter data for linear and non-linear decoder types
    decoder_df = df[df["decoder_type"] == "linear"]

    # Group by radius and pct_mask_modes, then compute mean & std across seeds for linear
    decoder_summary_df = decoder_df.groupby(["radius", "pct_mask_nodes"]).agg(
        mean_test_r2=(metric, "mean"),
        std_test_r2=(metric, "std"),
        mean_run_time=("runtime_seconds", "mean"),
        std_run_time=("runtime_seconds", "std"),
    ).reset_index()

    # Display the tables for linear and non-linear decoders
    print(f"{decoder_type} Decoder Summary ({metric}:")
    print(decoder_summary_df)
    return decoder_summary_df

def plot_robustness(df, metric="test_r2"):
    """
    Plots the robustness of a model's performance across different radii and 
    percentages of masked nodes.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing columns 'radius', 'pct_mask_nodes', and the specified metric.
    metric : str, optional (default="test_r2")
        The metric to plot on the y-axis. Can be "test_r2" for model performance 
        or "runtime_seconds" for computational cost.

    Returns:
    --------
    None
        Displays a line plot showing how the specified metric changes with radius 
        and percentage of masked nodes.

    Notes:
    ------
    - If metric is "test_r2", the y-axis is limited to [0, 1] and labeled "Mean Test R² Score".
    - If metric is "runtime_seconds", the y-axis is limited to [0, 1500] and labeled "Mean runtime in seconds".
    - The standard deviation is shown as a shaded region.
    """
    plt.figure(figsize=(4, 6))
    sns.lineplot(
        data=df,
        x="radius",
        y=metric,
        hue="pct_mask_nodes",
        marker="o",
        palette="coolwarm",
        errorbar=("sd")  # Adds standard deviation as shaded region
    )
    
    # Formatting
    plt.xlabel("Radius")
    plt.legend(title="Pct Mask Nodes")
    if metric == "runtime_seconds":
        plt.ylim(0, 5000)  # Set y-axis range
        plt.ylabel("Mean runtime in seconds")
    else:
        plt.ylim(0, 1)  # Set y-axis range
        plt.ylabel("Mean Test R² Score")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.show()

def plot_parameter_space(df: pd.DataFrame, metrics: str = 'test_r2', save_path: str = None):

    # Apply LOWESS smoothing
    smoothed = lowess(df[metrics], df['total_parameters'], frac=0.4)  # frac controls smoothness
    # Create scatterplot with trend line
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='total_parameters', y=metrics, data=df, label='Data Points')
    
    # Plot smoothed trend
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOWESS Curve')
    
    # Labels and title
    plt.xlabel("Total Parameters")
    plt.ylabel(f"{metrics}")
    plt.title(f"Trend of {metrics} with Increasing Parameters")
    
    if save_path is not None:
        plt.savefig(f'{save_path}_total_parameters_vs_{metrics}.jpg', dpi=1200)
    
    plt.show()