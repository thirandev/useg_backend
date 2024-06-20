# app/__init__.py
from flask import Flask

app = Flask(__name__)

# Import the main module to register the routes
import app.main
