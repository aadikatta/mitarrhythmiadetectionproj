# mitarrhythmiadetectionproj

Uses the MIT BIH dataset to successfully train a model that can diagnose a patient's arrhythmia condition by evaluating their ECG signal data.

Note: Lowering epochs can help with runtime, but reduce the accuracy of the model.

##############################################


Identify patients and classify beats as either having normal or abnormal.


Make datasets by compiling the first ECG signal from each patient, keeping only the heartbeats, and adding all of the ECG data for each patient to a prime X and y set.

Splits prime X and y into training and testing sets.

Trains a pipline with ConvLSTM2D, BatchNormalization, and Conv3D. Data must be flattened to fit the dimensional requirements of the Dense output layer.

Prints the accuracy of the model using several metrics:
- AUC: how good model is at distinguishing between 2 things
- Accuracy: proportion of true results to inputs
- Recall: ratio of correct predictions to all real positives
- Precision: ratio of correct predictions to positive predictions