
import os
import pickle
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm to show progress
from utils.utils import extract_features_from_image, perform_pca, train_svm_model

# Function to run inference and create the submission file
def run_inference(TEST_IMAGE_PATH, svm_model, k, SUBMISSION_CSV_SAVE_PATH):
    test_images = os.listdir(TEST_IMAGE_PATH)
    test_images.sort()

    image_feature_list = []

    # Wrap the loop with tqdm to show progress bar
    for test_image in tqdm(test_images, desc="Processing Images", unit="image"):
        path_to_image = os.path.join(TEST_IMAGE_PATH, test_image)
        image = cv2.imread(path_to_image)
        image_features = extract_features_from_image(image)
        image_feature_list.append(image_features)

    features_multiclass = np.array(image_feature_list)
    features_multiclass_reduced = perform_pca(features_multiclass, k)

    # Predict using the trained SVM model
    multiclass_predictions = svm_model.predict(features_multiclass_reduced)

    df_predictions = pd.DataFrame(columns=["file_name", "category_id"])

    for i in range(len(test_images)):
        file_name = test_images[i]
        new_row = pd.DataFrame({"file_name": file_name,
                                "category_id": multiclass_predictions[i]}, index=[0])
        df_predictions = pd.concat([df_predictions, new_row], ignore_index=True)

    df_predictions.to_csv(SUBMISSION_CSV_SAVE_PATH, index=False)
    print(f"Submission file saved as '{SUBMISSION_CSV_SAVE_PATH}'")

# Function to load the model and run the inference
def run_inference_task():
    # Define the paths using the current working directory
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
    BASE_PATH = "C:/Users/User/OneDrive/Desktop/Project ISM/Baselines/phase_1a"  # Change this path as per the actual folder structure
    SAVE_PATH = "C:/Users/User/OneDrive/Desktop/Project ISM/Baselines/phase_1a/submission/"  # Change if needed

    # Path to test images (this is the critical part of your request)
    TEST_IMAGE_PATH = "/tmp/data/test_images"  # Default path as per instructions
    SUBMISSION_CSV_SAVE_PATH = os.path.join(SAVE_PATH, "submission.csv")

    # Load the trained model weights
    MODEL_PATH = os.path.join(SAVE_PATH, "multiclass_model.pkl")

    # Load the model weights
    with open(MODEL_PATH, 'rb') as file:
        svm_model = pickle.load(file)
    
    TEST_IMAGE_PATH = r"C:\Users\User\OneDrive\Desktop\Project ISM\Baselines\phase_1a\images"  # Adjust as needed
    # Run inference and create the submission CSV
    k = 100  # PCA component size
    run_inference(TEST_IMAGE_PATH, svm_model, k, SUBMISSION_CSV_SAVE_PATH)

if __name__ == "__main__":
    run_inference_task()
