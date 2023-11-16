import warnings 
warnings.filterwarnings('ignore')
import tensorflow as tf 
import pandas as pd 
import numpy as np , time
import joblib 
input_dim = 29
THRESHOLD = 0.02 # found while testing the data 
RETRAIN_THRESHOLD = 50 # Set your desired retraining threshold
incorrect_predictions_count = 0
loaded_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)), 
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(2, activation='elu'),
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(input_dim, activation='elu')
])

loaded_model.compile(optimizer="adam", loss="mse", metrics=["acc"])
loaded_model.load_weights('autoencoder_best_weights.hdf5')
loaded_pipeline = joblib.load('your_pipeline.joblib')
given_data = pd.read_csv('testing_data.csv')
# Create a new DataFrame to store wrongly predicted data
wrongly_predicted_data = np.empty((1,29)) # shape of the X_test_transformed data 

for current_index in range(given_data.shape[0]):
    df = given_data.iloc[[current_index]]
    actual_answer = int(df['Class'])
    X_train = df.drop('Class', axis=1)
    X_train.columns = map(str.lower, X_train.columns)
    X_train['log10_amount'] = np.log10(X_train.amount + 0.00001)
    X_train = X_train.drop(['time', 'amount'], axis=1)
    X_train = X_train.values
    X_test_transformed = loaded_pipeline.transform(X_train)
    reconstructions = loaded_model.predict(X_test_transformed, verbose=0)
    mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)

    # Check if the prediction is incorrect
    if mse > THRESHOLD:
        print('Anomaly')
        given_answer = 1
    else:
        given_answer = 0
        print('Genuine')
    if actual_answer != given_answer:
      print('Sorry this was found wrong , in real implementation a human or any automation process should do it , waiting 3 sec to read the data ')
      print(X_train)
      time.sleep(7)
      incorrect_predictions_count += 1
      wrongly_predicted_data = np.append(wrongly_predicted_data, X_test_transformed, axis =0 )
    # Retrain the model if the incorrect predictions exceed the threshold
    if incorrect_predictions_count >= RETRAIN_THRESHOLD:
        print(f"Retraining the model after {RETRAIN_THRESHOLD} incorrect predictions.")
        
        def train_online(model, pipeline, new_data):
            preprocessed_data = pipeline.transform(new_data)
            model.fit(preprocessed_data, preprocessed_data, epochs=1, batch_size=256)
            loaded_model.load_weights('autoencoder_best_weights.hdf5')
            return model
        loaded_model = train_online(loaded_model, loaded_pipeline, wrongly_predicted_data)
        incorrect_predictions_count = 0
