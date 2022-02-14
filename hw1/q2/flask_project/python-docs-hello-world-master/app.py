
from flask import Flask
import pytz
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def hello():
    timeofday = datetime.now(pytz.timezone('US/Pacific')).hour
    if timeofday <= 6 : 
        name = 'Night'
    elif timeofday <= 12 : 
        name = 'Morning'
    elif timeofday <= 18 : 
        name = 'Aftenoon'
    else: 
        name = "Evening"         
    return ("Hello World! Good {0}".format(name))

