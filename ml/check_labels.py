"""Check what labels the model learned"""
import pickle

with open('ml/models/preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
    
print("Learned label encoders:")
for encoder_name, encoder in preprocessors['label_encoders'].items():
    print(f"\n{encoder_name}: {list(encoder.classes_)}")