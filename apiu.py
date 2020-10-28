from flask import Flask
from flask_restful import Api,Resource


class HelloWorld(Resource):
    def get(self):
        return {"data":"hello world"}



app = Flask(__name__)
api = Api(app)


api.add_resource(HelloWorld,"/hello")

if __name__ == "__main__":
    app.run(debug=True)