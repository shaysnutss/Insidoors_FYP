import json
from flask import Flask, request, jsonify
from flask_cors import CORS

import pc_clf
import building_clf
import proxy_clf


app = Flask(__name__)

CORS(app)


@app.route('/train_pc', methods=['GET'])
def train_pc():
    # Train classifier

    pc_clf.train()
    
    return jsonify({"message": "Complete"}), 200


@app.route('/classify_pc', methods=['POST'])
def classify_pc():
    # Process the data

    data_json = json.loads(request.data)
    result = pc_clf.classify(str(data_json))
    
    return jsonify({"message": result}), 200


@app.route('/train_building', methods=['GET'])
def train_building():
    # Train classifier

    building_clf.train()
    
    return jsonify({"message": "Complete"}), 200


@app.route('/classify_building', methods=['POST'])
def classify_building():
    # Process the data

    data_json = json.loads(request.data)
    result = building_clf.classify(str(data_json))
    
    return jsonify({"message": result}), 200


@app.route('/train_proxy', methods=['GET'])
def train_proxy():
    # Train classifier

    proxy_clf.train()
    
    return jsonify({"message": "Complete"}), 200


@app.route('/classify_proxy', methods=['POST'])
def classify_proxy():
    # Process the data

    data_json = json.loads(request.data)
    result = proxy_clf.classify(str(data_json))
    
    return jsonify({"message": result}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
