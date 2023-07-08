import os

import pandas as pd
from pycaret.classification import (compare_models, save_experiment,
                                    save_model, setup, tune_model)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from stqdm import stqdm


# Custom Transformer to convert data to floats for Shap compatibility
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.astype(float)
        return X_transformed


# AutoML Class for PyCaret Experiments
class AutoML:
    """
    
    Attributes:

    - df: (pd.DataFrame)
    - model: (top performing model)
    - target: (string)
    - train_data: (pd.DataFrame)
    - test_data: (pd.DataFrame)
    - experiment_name (string)

    Methods:

    - run_experiment(self)
        - Sets up a PyCaret classification experiment with an input dataframe
        - Log the experiment and use MLFlow to handle tracking artifacts for each stage
        - Compare_Models from PyCaret (Top 2)
        - Save Experiment
        - Save Models
    
    - retrieve_test_data(self)
        - Find the subfolder in mlruns/ folder where the test data can be found
        - Return the test data as a DataFrame
    
    - automate(self)
        - Runs the first two methods sequentially.
    
    
    """

    def __init__(self, df: pd.DataFrame, target: str, experiment_name: str):
        """
        Attributes:
            - df: (pandas DataFrame)
            - target: (string) - Target variable trying to be predicted
        """
        self.df = df
        self.target = target
        self.test_data = None
        
        # Turn experiment_name into an attribute
        self.experiment_name = experiment_name

    def run_experiment(self):
        asset_path = os.getcwd() + "/pycaret_assets"
        
        #Initialize progress bar
        total_steps = 4
        progress_bar = stqdm(total=total_steps)
        
        ## TODO: Find all cat variables in data, encode them, convert bools to floats
        
        # Start the setup function
        progress_bar.set_description('Setting up experiment... ')

        custom_transformer = CustomTransformer()

        # Create a pipeline with the custom transformer
        custom_pipeline = make_pipeline(custom_transformer)

        experiment = setup(data=self.df, target=self.target, normalize=True,
                            experiment_name=self.experiment_name,
                            # log_experiment=True,
                            # log_profile=True,
                            # log_data=True,
                            # log_plots=True,
                            # verbose=False,
                            # profile=True,
                            fix_imbalance=True,
                            custom_pipeline=[("to_float", custom_pipeline)],
                            custom_pipeline_position=-1
                        )
        self.ydata = experiment.y_test_transformed
        self.xdata = experiment.X_test_transformed
        
        progress_bar.update(1)
        
        # Run compare models
        progress_bar.set_description('Comparing models... ')
        best_models = compare_models(
            include=["dt", "rf", "lightgbm", "xgboost", "catboost", "et"],
            fold=10,
            # n_select=2
        )
        progress_bar.update(1)

        # Save Experiment
        save_experiment(f'{asset_path}/experiments/{self.experiment_name}.pkl')

        ## Append the tuned models into a list.
        ## Rank them and choose the one that has the highest metric of choice

        # Tune the top 2 performing models
        progress_bar.set_description('Tuning the model... ')
        self.tuned_model = tune_model(best_models, search_library='optuna',
                                      search_algorithm='tpe', optimize="AUC", n_iter=5)
        progress_bar.update(1)

        progress_bar.set_description('Experiment complete! ')
        self.model_name = type(self.tuned_model).__name__

        save_model(model=self.tuned_model, 
                   model_name=f"{asset_path}/models/{self.model_name}",
                   model_only=True
                   )
        
        save_experiment(f'{asset_path}/experiments/{self.experiment_name}.pkl')
        progress_bar.update(1)
        
        progress_bar.update(total_steps - progress_bar.n)
        progress_bar.close()
        
    def automate(self):
        self.run_experiment()
