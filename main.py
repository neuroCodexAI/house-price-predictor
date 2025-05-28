from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained pipeline
with open('RidgeModel.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Load cleaned data for choices (no index column)
data = pd.read_csv('Cleaned_data.csv', index_col=False)

# If for some reason 'Unnamed: 0' still sneaks in, drop it
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

locations = sorted(data['location'].unique())

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # parse inputs
        location = request.form['location']
        bhk      = float(request.form['bhk'])
        bath     = float(request.form['bath'])
        sqft     = float(request.form['total_sqft'])

        # build DataFrame exactly matching training features
        input_df = pd.DataFrame(
            [[location, sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        # predict
        pred = pipe.predict(input_df)[0]
        pred_rupees = pred * 100_000
        return str(round(pred_rupees, 2))

    except Exception as e:
        print("Error during prediction:", e)
        return f"Error in prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
