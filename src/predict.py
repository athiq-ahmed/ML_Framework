import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing

from GM.ML_Framework.src.dispatcher import MODELS

TEST_DATA = 'GM/ML_Framework/Input/test.csv'
MODEL = MODELS['randomforest']
df = pd.read_csv(TEST_DATA)
df.shape

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    # predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load("GM/ML_Framework/models/label_encoder.pkl")
        cols = joblib.load("GM/ML_Framework/models/train_df_columns.pkl")

        for c in cols:
            lbl = preprocessing.LabelEncoder()
            # df_test.loc[:, c] = lbl.fit_transform(df_test[c].values.tolist())
            df.loc[:, c] = lbl.fit_transform(df[c].values.tolist())

        clf = joblib.load("GM/ML_Framework/models/clf.pkl")
        cols = joblib.load("GM/ML_Framework/models/train_df_columns.pkl")

        df = df[cols]
        preds = clf.predict_proba(df)[:,1]
        print(preds)
        # predictions += preds

    # predictions /= 5
    preds /= 5
    sub = pd.DataFrame(np.column_stack((test_idx, preds)), columns = ["id", "target"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission['id'] = submission.id.astype(int)
    submission.to_csv("GM/ML_Framework/output/submission.csv" , index=False)

submission.shape