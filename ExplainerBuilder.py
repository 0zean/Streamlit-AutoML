from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from pycaret.classification import load_model


class ExplainerBuilder:
    """
    
    Attributes:

    - model_path: (path to model)
    - test_path: (path to test data)
    - target: (string)
    - title: (string)

    Methods:

    - build_explainer(self)
        - Sets up the classification Explainer Dashboard
        - Loads test data from mlruns folder and splits in to features, target
        - Loads the PyCaret model from given path
        - Returns the ExplainerDashboard
        
    - dashboard_run(self)
        - Runs the dashboard of the default port (8050)
    
    """
    def __init__(self, model_path, x_test, y_test, target: str, title: str):
        self.model_path = model_path
        self.x_test = x_test
        self.y_test = y_test
        self.target = target
        self.title = title
        
    def build_explainer(self):
        # Test Data
        # test_df = pd.read_csv(self.test_path)
        # test_df = test_df.iloc[:, 1:]
        
        # Features, Target
        # X_test = test_df.drop(self.target, axis=1)
        # y_test = test_df[self.target]
        
        # Load model
        model = load_model(self.model_path)
        
        # Setting up the Classifier Explainer
        classif_experiment = ClassifierExplainer(model,
                                                 self.x_test,
                                                 self.y_test,
                                                 shap='tree',
                                                 n_jobs=-1)

        # Dashboard
        explainer = ExplainerDashboard(classif_experiment,
                                       title=self.title)
        
        return explainer
        
    def dashboard_save(self):
        from os import getcwd
        self.build_explainer().to_yaml(getcwd()+"\\xdashboard\\dashboard.yaml",
                                       explainerfile=getcwd()+"\\xdashboard\\explainer.joblib",
                                       dump_explainer=True)
