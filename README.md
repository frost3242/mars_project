--------------------------Project Description-----------------------------------
This project implements a Speech Emotion Recognition system that classifies human emotions from audio files. We use the given dataset, which includes emotional speech and song audio samples from 24 actors.

Emotions covered:

Neutral
Calm
Happy
Sad
Angry
Fearful
Disgust
Surprised

The final model achieves high per-class and overall performance and is robust across different actors and modalities.


-----------------------Pre-processing Methodology----------------------------

Dataset Handling
Data Source: Audio_Song_Actors_01-24/ and Audio_Speech_Actors_01-24/

Emotion labels are parsed from file names using predefined RAVDESS encoding.

-----------Feature Extraction
Librosa is used to extract features:

MFCC (40)
Chroma
Zero-Crossing Rate
RMS Energy

All features are zero-padded/truncated to a fixed length of 216 frames.

--------- Dataset Balancing
Classes are upsampled to the maximum class count to ensure balance.

--------- Feature Scaling
Features are flattened and standardized using StandardScaler (fit on train set, applied on validation set).
Scaled data is reshaped back to its original shape for CNN compatibility.

--------- Label Encoding
LabelEncoder is used for encoding string emotion labels.
Encoded labels are converted to one-hot vectors for classification.


------------------------------------Model Pipeline-----------------------------------
-------------------Model Architecture (CNN)
Input → Conv1D (64) → BN → MaxPool → Dropout  
      → Conv1D (128) → BN → MaxPool → Dropout  
      → Conv1D (256) → BN → MaxPool → Dropout  
      → Flatten → Dense (256) → Dropout  
      → Dense (128) → Dropout  
      → Dense (output) with Softmax




-------------------Training Configuration--------------------------

Optimizer: Adam (learning rate = 0.001)
Loss: Categorical Crossentropy
Metrics: Accuracy
class Weights: Automatically computed to handle imbalance

Callbacks:
EarlyStopping (patience = 8)
ReduceLROnPlateau (patience = 4, factor = 0.5)

--------------------Model Files Saved--------------------
Trained model: trained4_model.h5
Label encoder: label_encoder.pkl
Scaler: scaler.pkl
Classes: classes.npy



------------------------------Evaluation Metrics---------------------
---------------Overall Results

Overall Accuracy: 92.19%
Macro F1 Score: 92.17%

----------Emotion-wise Accuracy

 angry: 94.67% accuracy
 calm: 93.42% accuracy
 disgust: 92.00% accuracy
 fearful: 94.67% accuracy
 happy: 88.16% accuracy
 neutral: 96.00% accuracy
 sad: 82.67% accuracy
 surprised: 96.00% accuracy


-------------Per-Actor Accuracy
Accuracy computed for each of the 24 actors
Helps ensure the model generalizes well across speakers

output:
Per-Actor Accuracy:
 Actor_01: 100.00% accuracy
 Actor_02: 100.00% accuracy
 Actor_03: 92.86% accuracy
 Actor_04: 86.67% accuracy
 Actor_05: 91.67% accuracy
 Actor_06: 94.12% accuracy
 Actor_07: 90.24% accuracy
 Actor_08: 77.27% accuracy
 Actor_09: 95.83% accuracy
 Actor_10: 85.00% accuracy
 Actor_11: 92.86% accuracy
 Actor_12: 83.33% accuracy
 Actor_13: 94.59% accuracy
 Actor_14: 85.00% accuracy
 Actor_15: 80.00% accuracy
 Actor_16: 96.00% accuracy
 Actor_17: 93.10% accuracy
 Actor_18: 94.12% accuracy
 Actor_19: 84.21% accuracy
 Actor_20: 90.48% accuracy
 Actor_21: 100.00% accuracy
 Actor_22: 100.00% accuracy
 Actor_23: 95.45% accuracy
 Actor_24: 94.12% accuracy


confusion_matrix.png – Saved plot

-----------How to Run
Install dependencies:
pip install -r requirements.txt

---------Run training script:
python train_model.py

