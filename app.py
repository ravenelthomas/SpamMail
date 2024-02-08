from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Charger les modèles
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

# Page d'accueil avec formulaire
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        email = request.form['email']
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        result = "Spam" if prediction[0] == 1 else "Non Spam"

    return render_template('index.html', result=result)

# API endpoint pour la prédiction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email = data['email']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    result = "Spam" if prediction[0] == 1 else "Non Spam"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)