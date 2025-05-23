import os
import webbrowser
from app import create_app

# Initialize the Flask application
app = create_app()

if __name__ == '__main__':
    # Only open the browser if running locally (not on Render or cloud)
    if os.environ.get("RENDER") is None:
        webbrowser.open("http://127.0.0.1:5000")
    
    # Start the Flask app
    app.run(debug=False, host="0.0.0.0", port=5000)
