import os,json,pickle,io,sys,glob
from utils import ScoringService as ss, FeatureEngineering as fe
import numpy as np,flask
import pandas as pd
import yaml
from flask import Flask, make_response, request, jsonify
import jwt,datetime
from datetime import datetime as dt
from functools import wraps 



list_of_files = glob.glob('model/DTR*') # * means all if need specific format then *.sav
latest_file = max(list_of_files, key = os.path.getctime)
preprocessor_path  = "/Users/akinwande.komolafe/Documents/Curacel/model/pipeline.pkl"
model_path = f'/Users/akinwande.komolafe/Documents/Curacel/{latest_file}'
config = yaml.safe_load(open("config/config.yml"))
numerical_features = config['input_num']

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'thisismykey'

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ss.ScoringService.get_model(model_path) is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='The App is healthy. loaded the model successfully', status=status, mimetype='application/json')


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'error':'token is missing. Set key to Token'}), 401
        try:
            jwt.decode(token, app.config['SECRET_KEY'])
        except:
            return jsonify({'error': 'Invalid token or Token expired'}), 401
            
        return f(*args, **kwargs)
    return decorated


@app.route('/inference', methods=['POST'])
@token_required
def predict():
    df = flask.request.get_json(force=True)
    data = pd.io.json.json_normalize(df)
    for col in numerical_features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    feat = fe.feature_engineering(data,numerical_features)
    new_df = feat._preprocessing(train=False,target=False,predict=True)
    if 'price' in new_df.columns:
        new_df.drop(['price'],inplace=True,axis=1)
    preprocessor = ss.ScoringService.get_preprocessor(preprocessor_path)
    new_df = preprocessor.transform(new_df)
    predictions = ss.ScoringService.predict(new_df,model_path)
    out = io.StringIO()
    pd.DataFrame({'results':predictions.flatten()}).to_csv(out, index=False,sep=',',header=['price of car (PHP)'])
    result = out.getvalue()
    return result



@app.route('/login')
def login():
    auth = request.authorization
    
    if auth and auth.password == 'Curacel<>?':
        token = jwt.encode({'user': auth.username, 'exp':dt.utcnow() + datetime.timedelta(minutes=10)},\
                           app.config['SECRET_KEY'])
        return jsonify({'token':token.decode('UTF-8')})
        
    return make_response('could not verify!, Login is Required',401,{'WWW-Authenticate':'Basic realm = "Login Required"'})

if __name__ == "__main__":
    app.run(host ='0.0.0.0',port=5500, debug=True)