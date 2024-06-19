import pandas as pd
from flask import Flask, render_template, request
import main
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', data_req='/static/image/data_req.png')

@app.route('/about')
def about():
    return render_template('about.html', evaluations={})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    datafile = request.files['datafile']
    data_path = './static/' + datafile.filename
    datafile.save(data_path)

    # KMeans
    kmeans = main.k_means(path=data_path, outliers_fraction=0.01)
    kmeans.pre_processing()
    data = kmeans.standarizing()
    model, data = kmeans.training(data=data)
    kmeans.predict(K_means=model, data=data)
    kmeans.visualisation()
    kmeans_result = kmeans.evaluate(data)

    # GaussianMixture
    gaussian_mix = main.gaussian(path=data_path, outliers_fraction=0.01, support_fraction=0.999)
    gaussian_mix.pre_processing()
    gaussian_mix.predict()
    gaussian_mix.visualisation()
    gaussian_result = gaussian_mix.evaluate(data)

    # IsolationForest
    isoForest = main.isolationForest(path=data_path, outliers_fraction=0.01)
    isoForest.pre_processing()
    data = isoForest.standarizing()
    model, data = isoForest.training(data)
    isoForest.predict(model, data)
    isoForest.visualisation()
    iso_result = isoForest.evaluate(data)

    # UnsupervisedKNN
    knn = main.unsup_knn(path=data_path, threshold=1.5)
    knn.pre_processing()
    data = knn.standarizing()
    model, data = knn.training(data=data)
    knn.predict(knn=model, data=data)
    knn.visualisation()
    knn_result = knn.evaluate(data)

    evaluations = {
        "kmeans": kmeans_result["evaluation"],
        "gaussian": gaussian_result["evaluation"],
        "isoforest": iso_result["evaluation"],
        "unsupKNN": knn_result["evaluation"]
    }

    anomalies = {
        "kmeans": kmeans_result["anomalies"],
        "gaussian": gaussian_result["anomalies"],
        "isoforest": iso_result["anomalies"],
        "unsupKNN": knn_result["anomalies"]
    }

    return render_template('results.html', flag=True, evaluations=evaluations, anomalies=anomalies)

if __name__  == '__main__':
    app.run(port=5000, debug=True)