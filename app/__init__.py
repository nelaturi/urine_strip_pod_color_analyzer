from flask import Flask
from flask_cors import CORS
import os

def create_app():
    # Initialize Flask app
    app = Flask(__name__)

    # Define the folder where uploaded and output images are stored
    app.config['UPLOAD_FOLDER'] = os.path.join("app", "static", "uploads")

    # Secret key for session handling, flash messages, etc.
    app.secret_key = 'urine-strip-secret-key'

    # Allow CORS for API use from mobile apps / frontend clients
    CORS(app)

    # Register route blueprint
    from .routes import main
    app.register_blueprint(main)

    return app
