from flask import Flask,render_template,request
import pandas as pd
import pickle
app = Flask(__name__)

data = pd.read_csv("cleaned_data.csv")
model = pickle.load(open('banglore_home_prices_model.pkl','rb'))

@app.route("/")
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)


@app.route("/predict",methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('sqft'))

    print(location,"--" , bhk,"--" , bath,"--" , sqft)

    input_data = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = model.predict(input_data)[0]

    

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True,port=5000)