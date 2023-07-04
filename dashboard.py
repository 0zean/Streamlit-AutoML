from os import getcwd

from explainerdashboard import ExplainerDashboard

db = ExplainerDashboard.from_config(getcwd()+"/xdashboard/dashboard.yaml")
app = db.flask_server()
