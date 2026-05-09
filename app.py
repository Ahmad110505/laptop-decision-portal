from flask import Flask

from routes.home_routes import home_bp
from routes.prediction_routes import prediction_bp
from routes.fuzzy_routes import fuzzy_bp
from routes.anomaly_routes import anomaly_bp
from routes.rl_routes import rl_bp
from routes.admin_routes import admin_bp


app = Flask(__name__)
app.secret_key = "laptop_decision_portal_secret_key"

app.register_blueprint(home_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(fuzzy_bp)
app.register_blueprint(anomaly_bp)
app.register_blueprint(rl_bp)
app.register_blueprint(admin_bp)


if __name__ == "__main__":
    app.run(debug=True)