import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib

# from . import dispatcher
# from src.dispatcher import MODELS
from GM.ML_Framework.src.dispatcher import MODELS
import matplotlib.pyplot as plt

# os.getcwd()
# TRAINING_DATA = os.environ.get("TRAINING_DATA")
# FOLD = int(os.environ.get("FOLD"))
# MODEL = os.environ.get("MODEL")

TRAINING_DATA = 'GM/ML_Framework/Input/train_folds.csv'
TEST_DATA = 'GM/ML_Framework/Input/test.csv'
FOLD = int(0)
MODEL = MODELS['randomforest']
# print(MODELS.keys())


FOLD_MAPPING = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [2,3,4,1],
    3 : [4,2,1,0],
    4 : [1,2,3,0]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        # lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.fit_transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.fit_transform(valid_df[c].values.tolist())
        # df_test.loc[:, c] = lbl.fit_transform(df_test[c].values.tolist())
        label_encoders[c] = lbl

    # Data is ready to train
    clf = MODEL
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    # print(preds)
    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders, "GM/ML_Framework/models/label_encoder.pkl")
    joblib.dump(clf, "GM/ML_Framework/models/clf.pkl")
    joblib.dump(train_df.columns, "GM/ML_Framework/models/train_df_columns.pkl")

