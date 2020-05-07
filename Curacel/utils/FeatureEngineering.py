import pandas as pd
import numpy as np
#from sklearn.neighbors import KNeighborsRegressor
import pickle

class feature_engineering:
    """ Apply reusable feature engineering functions """
    def __init__(self,data,numerical_features=None,categories=None,categorical_features=None):
        self.data = data
        self.numerical_features = numerical_features
        self.categories = categories
        self.hash_features = categorical_features
        
    
    def _preprocessing(self,train=True,target=True, predict=True):
        self.data['age_of_car'] = 2019 - self.data['Year of Manufacture']
        #deleted Poster, Version and Description
        drop_col = ['Poster','Version','Description','Title','Used','name','Date Posted','Year of Manufacture']
        self.data.drop(columns=drop_col,inplace=True,axis=1)
        if train:
            self.data = self.data[self.data['age_of_car']>=0]
            self.data = self.data[self.data['retail']>0]
            self.data = self.data[self.data['Color Family'].notnull()]
            self.data = self.data[self.data['Fuel Type'].notnull()]
            self.X = pd.DataFrame(self.data['age_of_car'].loc[self.data['Mileage (in km)'].notnull()])
            self.X_null = pd.DataFrame(self.data['age_of_car'].loc[self.data['Mileage (in km)'].isnull()])
            self.y = pd.DataFrame(self.data['Mileage (in km)'].loc[self.data['Mileage (in km)'].notnull()])
            self.y_null = pd.DataFrame(self.data['Mileage (in km)'].loc[self.data['Mileage (in km)'].isnull()])
            self.clf = KNeighborsRegressor(3,weights='distance')
            self.trained_model = self.clf.fit(self.X,self.y)
            self.imputed_val = pd.DataFrame(self.trained_model.predict(self.X_null),columns=['Mileage (in km)'])
            self.X_null.reset_index(inplace=True)
            pickle.dump(self.trained_model, open('model/KNR.sav', 'wb'))
            
        if predict:
            self.X = pd.DataFrame(self.data['age_of_car'].loc[self.data['Mileage (in km)'].notnull()])
            self.X_null = pd.DataFrame(self.data['age_of_car'].loc[self.data['Mileage (in km)'].isnull()])
            self.y = pd.DataFrame(self.data['Mileage (in km)'].loc[self.data['Mileage (in km)'].notnull()])
            self.y_null = pd.DataFrame(self.data['Mileage (in km)'].loc[self.data['Mileage (in km)'].isnull()])
            with open('model/KNR.sav', 'rb') as file:
                self.trained_model = pickle.load(file)
            self.imputed_val = pd.DataFrame(self.trained_model.predict(self.X_null),columns=['Mileage (in km)'])
            self.X_null.reset_index(inplace=True)
        
        self.null_merged = pd.merge(self.X_null, self.imputed_val, how='outer',on = self.X_null.index)
        self.null_merged.drop('key_0',inplace=True, axis=1)
        self.null_merged.set_index('index',inplace=True)
        
        self.xy_merged = pd.concat([self.X,self.y],axis=1)
        self.merged_df = pd.concat([self.null_merged,self.xy_merged],axis=0)
        self.merged_df = self.merged_df.sort_index()
        
        self.total_merge = pd.merge(self.data, self.merged_df, on= self.data.index)
        self.total_merge.drop(['age_of_car_x','Mileage (in km)_x','key_0'],axis=1,inplace=True)
        col = ['id','price','brand','model','transmission','color','body_type','poster_type','fuel_type','location','retail','post_age_in_days','age_of_car','mileage_in_km']
        self.total_merge.columns = col
        if target:
            self.total_merge['price'] = np.log(self.total_merge['price'])
        
        return self.total_merge
        
    def _train_test_split(self,test_size):
        self._preprocessing()
        self.train, self.val = train_test_split(self.total_merge, test_size=test_size, random_state=4)
        self.ytrain = self.train['price']
        self.ytest = self.val['price']
        self.val.drop(['price'],axis=1, inplace=True)
        self.train.drop(['price'],axis=1,inplace=True)
        
    
    def _build_pipeline(self,hash_vector_size):
        self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', StandardScaler())])
        self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                               ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
        self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                          ('hasher', FeatureHasher(n_features=hash_vector_size, input_type='string'))])
        
        
    def fit_pipeline(self,test_size=0.2,hash_vector_size=15):
        self._train_test_split(test_size)
        self._build_pipeline(hash_vector_size)
        print('Pipeline built successfully')
        self.full_pipeline = ColumnTransformer([
                                ("num", self.num_pipeline, self.numerical_features),
                                ("cat", self.cat_pipeline, self.categories),
                                ("hash", self.hash_pipeline, self.hash_features)])
        self.full_pipeline.fit(self.train)
        self.preprocessed = self.full_pipeline.transform(self.train)
        self.test = self.full_pipeline.transform(self.val)
        print('Data preprocessed successfully')
        return self.preprocessed, self.test, self.ytrain, self.ytest
    
    def save_pipeline(self,model=None,model_name=None,pipeline_name =None):
        # save the model to disk
        pickle.dump(model, open(model_name, 'wb'))
        
        
        if pipeline_name:
            # save the pipeline to disk
            pickle.dump(self.full_pipeline, open(pipeline_name, 'wb'))