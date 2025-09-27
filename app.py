from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from src.logger import logging

application = Flask(__name__)
app = application   

##   Route for home page

@app.route('/')
def index():
    logging.info("We are in index page")
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info(("We are in home page"))
        return render_template('home.html')
    else:
        logging.info("We are in home page and we are predicting")
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        pred_df = data.get_data_as_data_frame()
        logging.info(f"Dataframe for prediction: {pred_df}")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info(f"Prediction results: {results}")
        return render_template('home.html', results=results)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) 
    app.run(debug=True)


