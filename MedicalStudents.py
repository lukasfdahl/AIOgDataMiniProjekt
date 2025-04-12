import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # enable experimental
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import SQLSystem as sql


def main():
    #--------------------------------#
    #---------- Imputation ----------#
    #--------------------------------#
    dataset = pd.read_csv("datasæt/medical_students_dataset.csv")
    print(dataset)
    dataset = preprocess_student_id(dataset) #fill missing student id's
    dataset = preprocess_height_weight_bmi(dataset) #calculate BMI, Height, Weight from each other

    #here i use encoding to make sure everything is numbers, so that i later can use regression impuation to find the missing values
    #the following featues only have 2 options so they can be easilly converted to numbers with oridnal encoding
    smoking_enc = OrdinalEncoder()
    diabetes_enc = OrdinalEncoder()
    gender_enc = OrdinalEncoder()

    dataset["Smoking"] = smoking_enc.fit_transform(dataset[["Smoking"]])
    dataset["Diabetes"] = diabetes_enc.fit_transform(dataset[["Diabetes"]])
    dataset["Gender"] = gender_enc.fit_transform(dataset[["Gender"]])

    #here i remove blood type so that it dosent get imputated with the rest using regression. instead i will use randomforest on it later
    dataset_blood_type = dataset[["Blood Type"]]
    dataset = dataset.drop(columns=["Blood Type"])

    #here i use regression imputation to fill the remaning missing values, using sklearns itterative imupter and bayesianRidge as the regression model
    #bayesianRidge should be more stable than linear rgeression and handle highly correlated features better
    imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0)
    dataset =   pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

    #randomforest imputation for blood type.
    blood_enc = OrdinalEncoder()
    dataset_blood_type["Blood Type"] = blood_enc.fit_transform(dataset_blood_type[["Blood Type"]])
    imputer = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=0)
    imputed_blood = imputer.fit_transform(dataset_blood_type[["Blood Type"]])
    dataset_blood_type["Blood Type"] = blood_enc.inverse_transform(imputed_blood).ravel()

    dataset = pd.concat([dataset, dataset_blood_type], axis=1)

    print(dataset)
    

    #---------------------------------#
    #-------------- SQL --------------#
    #---------------------------------#
    #to show the usage of SQL i create 2 tables. one with all the male students and one with all the female students
    database = sql.create_connection("students.db")
    
    #create the table for students
    sql.execute_query(database, """CREATE TABLE IF NOT EXISTS Female_Students (
        Student_ID INT,
        Age FLOAT,
        Height FLOAT,
        Weight FLOAT,
        Blood_Type TEXT,
        BMI FLOAT,
        Temperature FLOAT,
        Heart_Rate FLOAT,
        Blood_Pressure FLOAT,
        Cholesterol FLOAT,
        Diabetes TINYINT,
        Smoking TINYINT);""")
    
    #the qurry used to insert data into the table
    insert_query = f"""
        INSERT OR IGNORE INTO Female_Students (
            Student_ID, Age, Height, Weight, Blood_Type,
            BMI, Temperature, Heart_Rate, Blood_Pressure, Cholesterol,
            Diabetes, Smoking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
    
    #get all the values that need to be inserted
    dataset_f = dataset[dataset["Gender"] == 0]
    values_to_insert = dataset_f[["Student ID", "Age", "Height", "Weight", "Blood Type", "BMI", "Temperature", "Heart Rate", "Blood Pressure", "Cholesterol", "Diabetes", "Smoking"]].values.tolist()
    sql.execute_query(database, insert_query, values_to_insert)


    #for male students:
    #create the table for students
    sql.execute_query(database, """CREATE TABLE IF NOT EXISTS Male_Students (
        Student_ID INT,
        Age FLOAT,
        Height FLOAT,
        Weight FLOAT,
        Blood_Type TEXT,
        BMI FLOAT,
        Temperature FLOAT,
        Heart_Rate FLOAT,
        Blood_Pressure FLOAT,
        Cholesterol FLOAT,
        Diabetes TINYINT,
        Smoking TINYINT);""")
    
    #the qurry used to insert data into the table
    insert_query = f"""
        INSERT OR IGNORE INTO Male_Students (
            Student_ID, Age, Height, Weight, Blood_Type,
            BMI, Temperature, Heart_Rate, Blood_Pressure, Cholesterol,
            Diabetes, Smoking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
    
    #get all the values that need to be inserted
    dataset_m = dataset[dataset["Gender"] == 1]
    values_to_insert = dataset_m[["Student ID", "Age", "Height", "Weight", "Blood Type", "BMI", "Temperature", "Heart Rate", "Blood Pressure", "Cholesterol", "Diabetes", "Smoking"]].values.tolist()
    sql.execute_query(database, insert_query, values_to_insert)
    print("Done Creating SQL")

#prints the number of na values in a given feature
def print_na_count(dataset : pd.DataFrame, name : str):
    print(f"{name}-NA = {len(dataset[name][dataset[name].isna()])}")


def preprocess_student_id(dataset : pd.DataFrame):
    #student id just goes up one for each row, resetting halfway thrugh, so it can easily be imputated with the following:
    dataset["Student ID"] = (dataset.index % 100000) + 1
    return dataset

def preprocess_height_weight_bmi(dataset : pd.DataFrame):
    #if only one of BMI, Height, Weight is missing, the remaning one can be calculated with the formular for bmi
    dataset["BMI"] = dataset.apply(lambda x: x["Weight"] / ((x["Height"]/100)**2) if pd.notna(x["Height"]) and pd.notna(x["Weight"]) and pd.isna(x["BMI"]) else x["BMI"], axis=1)
    dataset["Height"] = dataset.apply(lambda x: np.sqrt(x["Weight"] / x["BMI"])*100 if pd.isna(x["Height"]) and pd.notna(x["Weight"]) and pd.notna(x["BMI"]) else x["Height"], axis=1)
    dataset["Weight"] = dataset.apply(lambda x: x["Weight"] * (x["Height"]/100)**2 if pd.notna(x["Height"]) and pd.isna(x["Weight"]) and pd.notna(x["BMI"]) else x["Weight"], axis=1)
    return dataset





if __name__ == "__main__":
    main()