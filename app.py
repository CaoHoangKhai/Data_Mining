from flask import Flask, render_template, request, flash
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
models = {
    'KNN': pickle.load(open('knn_model.pkl', 'rb')),
    'Naive Bayes': pickle.load(open('nb_model.pkl', 'rb')),
    'Decision Tree': pickle.load(open('dt_model.pkl', 'rb'))
}
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Retrieve form data
            area = float(request.form['area'])
            perimeter = float(request.form['perimeter'])
            major_axis_length = float(request.form['major_axis_length'])
            minor_axis_length = float(request.form['minor_axis_length'])
            eccentricity = float(request.form['eccentricity'])
            convex_area = float(request.form['convex_area'])
            extent = float(request.form['extent'])

            # Get selected algorithm
            algorithm = request.form['algorithm']

            # Prepare input data
            input_data = [[area, perimeter, major_axis_length, minor_axis_length, eccentricity, convex_area, extent]]

            # Predict using the selected model
            prediction = models[algorithm].predict(input_data)[0]
        except KeyError as e:
            flash(f"Missing input field: {str(e)}. Please fill in all fields.")
        except ValueError:
            flash("Please enter valid numerical values for all fields.")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
