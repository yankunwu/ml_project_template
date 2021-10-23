from sklearn import preprocessing
import pandas as pd
import os
from sklearn import ensemble
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher



TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

def predict():
    test_df = pd.read_csv(TEST_DATA)
    test_idx = test_df["id"].values
    predictions = None

    for FOLD in range(5):
        df = test_df
        encoders = joblib.load(os.path.join("models", f"{MODEL}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))

        for c in cols:
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
            

        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])

    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)