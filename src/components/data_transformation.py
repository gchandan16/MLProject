import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os

#comlumnTransformer basically use for pipline
from sklearn.compose  import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

#for require exception log
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        # this below function responsible for data transformation

    def get_data_transformer_object(self):
        try:
              
            # this use fornumerical feature
            numerical_column=["writing_score","reading_score"]
            #this use for statical feature
            categorial_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]


            # create two pipline
            #steps1 imputer handle the missing value

            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("scaler",StandardScaler())
                       ]
            )

            cat_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")), 
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standards scaling completed");
            logging.info("Categorial columns encoding completed") 

            # Basically transformer is combination two numerical & categorial pipline

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipline",cat_pipeline,categorial_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error(f"Error occurred while get_data_transformer_object: {e}")
            raise CustomException(e,sys)  
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed ")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing Objects")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"Error occurred while reading the dataset: {e}")
            raise CustomException(e,sys)
            


