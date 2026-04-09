"""
WSGI entry point for PythonAnywhere
Configure in PythonAnywhere Web tab:
  Source code: /home/yourusername/okx_bot_test
  Working directory: /home/yourusername/okx_bot_test
  WSGI configuration file: /var/www/yourusername_pythonanywhere_com_wsgi.py
"""

import os
import sys

# Add project directory to Python path
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import Flask app
from web_dashboard import app as application

# Enable debug mode (disable in production)
application.debug = False
