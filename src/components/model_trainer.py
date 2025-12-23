import os
import sys
from dataclasses import dataclass
from catboost import catBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor
                              , GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_file=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config=ModelTrainerConfig

    def initiate_model_trainer(self,train_array,test_array):
        try :
            logging.info(" split training and testing input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
            
                "Random Forest":RandomForestRegressor(),
                " Decision Tree": DecisionTreeRegressor(),
                "gradient bossting ": GradientBoostingRegressor(),
                "Linear regression":LinearRegression(),
                " K-Neighbors classifier": KNeighborsRegressor(),
                " XGB classifier": XGBRegressor(),
                " Catboosting classifier": catBoostRegressor(),
                "Adaboost classifier": AdaBoostRegressor(),



            }
            

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train , x_test=x_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_maodel_name=list(model_report.keys())[

                list(model_report.values()).index(best_model_score)

            
            ]

            best_model=models[best_maodel_name]

            if best_model_score<0.6:
                raise CustomException(" no best model found")
            
            logging.info(" best found model on both training and testing")

            save_object(
                file_path=self.Model_trainer_config.trained_model_file_path,
                obj= best_model

            )

            predicted=best_model.predect(x_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)

            