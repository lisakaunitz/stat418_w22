
# In[7]:


# Imports
import pandas as pd
import numpy as pd
import os
import sqlite3
import requests
import json
import sqlalchemy as db
from flask import Flask, escape, request, Response, render_template
from datetime import datetime
# pip install Levenshtein
# pip install dicttoxml
from dicttoxml import dicttoxml # Converts a Python dictionary or other native data type into a valid XML string.
from Levenshtein import distance # https://www.statology.org/levenshtein-distance-in-python/


# In[13]:


app = Flask(__name__)
@app.route('/')
def func(row):
    xml = ['<item>']
    for field in row.index:
        xml.append('  <field name="{0}">{1}</field>'.format(field, row[field]))
    xml.append('</item>')
    return '\n'.join(xml)
 
    # Part 1
@app.route("/find/", methods=["GET"])
def movie_scout():
    query = request.args.get("foo")
    
    # api
    url = "http://www.omdbapi.com/?s={0}".format(query)
    r = requests.get(url+"&apikey=5c2a390c")
    omdb_return = json.loads(r.text)

    
    if omdb_return['Response'] == "True":
        
        dists = [distance(s['Title'], query) for s in omdb_return['Search']]
        best_return = omdb_return['Search'][np.argmin(dists)]['Title'].lower()

        # connecting to the sqlite server
        engine = db.create_engine('sqlite:///../q1/movieratings.db')
        connection = engine.connect()
        metadata = db.MetaData()
        movies = db.Table('movies', metadata, autoload=True, autoload_with=engine)
     
        # query sqlite
        query = db.select([movies]).where(db.func.lower(movies.columns.movie_title) == '{0}'.format(best_return)).limit(1)
        ResultProxy = connection.execute(query)
        col = ResultProxy.keys()
        ResultSet = pd.DataFrame(ResultProxy.fetchall(), columns=col)
        if len(ResultSet) == 0:
            return "ERROR 404", 404
        
        # Return xml
        xml = dicttoxml(ResultSet.iloc[0,:].to_dict())
        return(Response(response=xml, status = 200, mimetype = "application/xml"))
        
    else:
        return "404 Error Code", 404
    
    
    # Part 2
@app.route('/add/', methods=["GET", 'POST', "PUT"])
def num_two_add():
    if request.method == "GET" or request.method == "PUT":
        return "404 Error Code", 404
    else:
        input_json = dict()
        input_json['username'] = request.form['username']
        input_json['movie'] = request.form['movie']
        input_json['rating'] = request.form['rating']
        conn = sqlite3.connect("../q1/movieratings.db") # connect to sqlite
        cursor = conn.cursor()
        print("connected to sqlite")

 
        # insert the user
        try:
            sqlite_query = """INSERT INTO users
                          (user_name, first_name, last_name) 
                          VALUES (?, ?);"""
            recordList = [[input_json['username'],input_json['username'], '']]
            cursor.executemany(sqlite_query, recordList)
            conn.commit()
            conn.close()
        except:
            pass
 
        # insert the rating
        conn = sqlite3.connect("../q1/movieratings.db")
        cursor = conn.cursor()
        sqlite_query = """INSERT INTO ratings
                          (rating) 
                          VALUES (?);"""
        recordList = [input_json['rating']]
        cursor.execute(sqlite_query, recordList)
        conn.commit()
        conn.close()
        
        # insert the movie
        conn = sqlite3.connect("../q1/movieratings.db")
        cursor = conn.cursor()
        sqlite_query = """INSERT INTO movies
                          (movie_title) 
                          VALUES (?);"""
        recordList = [input_json['movie'],]
        cursor.execute(sqlite_query, recordList)
        conn.commit()
        conn.close()
        return('Congrats, your rating was added')
 

 
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




