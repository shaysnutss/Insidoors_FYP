import json
import socket
import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import json

import pc_clf
import building_clf
import proxy_clf

# my_id = os.getenv('db_conn')
app = Flask(__name__)
string1 = 'mysql+mysqlconnector://root:'
string2 = 'akshaya100@host.docker.internal:3306/food'
app.config['SQLALCHEMY_DATABASE_URI'] = string1 + string2


# if db_urls:
#     app.config['SQLALCHEMY_DATABASE_URI'] = db_urls + '/food'
# else:
#     app.config['SQLALCHEMY_DATABASE_URI'] = None
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 100,
                                           'pool_recycle': 280}
db = SQLAlchemy(app)

CORS(app)


class Food(db.Model):
    __tablename__ = 'food'

    food_id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(64), nullable=False)
    category = db.Column(db.String(64), nullable=False)

    def __init__(self, food_name, category):
        self.food_name = food_name
        self.category = category

    def to_dict(self):
        return {
            'food_id': self.food_id,
            'food_name': self.food_name,
            'category': self.category
        }

# db.create_all()

# to populate the food table with data

# table_row  = Food('Burger', 'Fast Food')
# db.session.add(table_row)
# db.session.commit()


@app.route("/health")
def health_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is healthy.",
                "service:": "food",
                "ip_address": local_ip
            }
    ), 200


@app.route("/food")
def get_all():
    food_list = Food.query.all()
    if len(food_list) != 0:
        return jsonify(
            {
                "data": {
                    "food": [food.to_dict() for food in food_list]
                }
            }
        ), 200
    return jsonify(
        {
            "message": "There are no foods."
        }
    ), 404


@app.route("/food/<int:food_id>")
def find_by_id(food_id):
    food = Food.query.filter_by(food_id=food_id).first()
    if food:
        return jsonify(
            {
                "data": food.to_dict()
            }
        ), 200
    return jsonify(
        {
            "message": "Food not found."
        }
    ), 404


@app.route("/food/<int:food_id>", methods=['DELETE'])
def delete_by_id(food_id):
    food = Food.query.filter_by(food_id=food_id).first()
    if food:
        try:
            db.session.delete(food)
            db.session.commit()
            return jsonify(
                {
                    "data": {
                        "food_id": food_id
                    }
                }
            ), 200
        except Exception as e:
            return jsonify(
                {
                    "message": "An error occurred deleting the food.",
                    "error": str(e)
                }
            ), 500
    return jsonify(
        {
            "data": {
                "food_id": food_id
            },
            "message": "Food not found."
        }
    ), 404



#CHANGE , check if food present in db
@app.route("/food", methods=['POST'])
def new_food():
    #check if food present
    try:
        data = request.get_json()
        food = Food(**data)
        food_name = food.food_name
        food1 = Food.query.filter_by(food_name=food_name).first()
        if food1:
            food_idy = food1.food_id
            return jsonify(
                {
                    "food_id": food_idy
                }
            ), 201
        else:
            db.session.add(food)
            db.session.commit()
            return jsonify(
                {
                    "food_id": food.to_dict().get('food_id')
                }
            ), 201
    except Exception as e:
        return jsonify(
            {
                "message": "An error occurred creating the food.",
                "error": str(e)
            }
        ), 500



@app.route("/food/<int:food_id>", methods=['PUT'])
def change_by_id(food_id):
    food = Food.query.filter_by(food_id=food_id).first()
    if food:
        try:
            data = request.get_json()
            food1 = Food(**data)
            food.food_name = food1.food_name
            food.category = food1.category
            db.session.commit()
            return jsonify(
                {
                    "food_id": food.to_dict().get('food_id')
                }
            ), 200
        except Exception as e:
            return jsonify(
                {
                    "message": "An error occurred replacing the food.",
                    "error": str(e)
                }
            ), 500
    return jsonify(
        {
            "data": {
                "food_id": food_id
            },
            "message": "Food not found."
        }
    ), 404

@app.route('/example_pc', methods=['POST'])
def example_pc():
    # Process the data

    data_json = json.loads(request.data)
    result = pc_clf.infer(str(data_json))
    
    return jsonify({"message": result}), 200

@app.route('/example_building', methods=['POST'])
def example_building():
    # Process the data

    data_json = json.loads(request.data)
    result = building_clf.infer(str(data_json))
    
    return jsonify({"message": result}), 200

@app.route('/example_proxy', methods=['POST'])
def example_proxy():
    # Process the data

    data_json = json.loads(request.data)
    result = proxy_clf.infer(str(data_json))
    #result = result.replace('"', "'")
    
    return jsonify({"message": result}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
