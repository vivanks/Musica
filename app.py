from flask import Flask, render_template,flash,url_for,session,logging,request,redirect
from flask import request, jsonify
import os


app = Flask(__name__)
app.secret_key='secret123'

import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation

#Read userid-songid-listen_count triplets
#This step might take time to download data from external sources
triplets_file = '10000.txt'
songs_metadata_file = 'song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 

song_df = song_df.head(10000)

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

users = song_df['user_id'].unique()
songs = song_df['song'].unique()

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')



@app.route('/',methods=['GET','POST'])
def index():
	if request.method == 'POST':
		#Get form field
		mid = request.form['mid']
		pas = request.form['pass']

		
		song_name = mid+" - "+pas

		predicted_song = is_model.get_similar_items([song_name])
		
		return render_template('values.html',predicted_song=predicted_song["song"])
	return render_template('index.html')

@app.route('/all')
def all():
	return render_template('all.html',song = song_df['song'])

if __name__ =="__main__":
	
	app.run(debug = True)