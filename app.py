from src.RestaurantRatingPrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(

            online_order = object(request.form.get('online_order')),
            book_table = object(request.form.get('book_table')),
            votes = int(request.form.get('votes')),
            location = object(request.form.get('location')),
            rest_type = object(request.form.get('rest_type')),
            cuisines = object(request.form.get('cuisines')),
            cost_for_2 = object(request.form.get('cost_for_2')),
            type = object(request.form.get('type'))
        )

        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8090)


    