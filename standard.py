import cv2
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess images for HOG features
def preprocess_images(images):
    hog_features = []
    for image in tqdm(images):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_feature = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(hog_feature)
    return np.array(hog_features)

x_train_hog = preprocess_images(x_train)
x_test_hog = preprocess_images(x_test)

# Define and train SVM classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(0.8)),
    ('svm', SVC(kernel='rbf', C=10, cache_size=10000))
])
pipe.fit(x_train_hog, y_train)

# Evaluate classifier
accuracy = pipe.score(x_test_hog, y_test)
print("Test Accuracy:", accuracy)

# Predict labels for test data
y_pred = pipe.predict(x_test_hog)

# Generate and display classification report
report = classification_report(y_test, y_pred, target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
print(report)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Visualize random images with predicted labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

for i in range(5):
    for j in range(5):
        random_index = random.randint(0, len(x_test) - 1)
        sample_image = x_test[random_index]
        true_class = y_test[random_index]
        true_class_label = class_labels[true_class[0]]  # Extract scalar integer class label
        gray_sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
        hog_features = hog(gray_sample_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features_reshaped = hog_features.reshape(1, -1)
        predicted_class = pipe.predict(hog_features_reshaped)[0]
        predicted_class_label = class_labels[predicted_class]
        axes[i, j].imshow(sample_image)
        axes[i, j].axis('off')
        axes[i, j].set_title(f'Actual: {true_class_label}\nPredicted: {predicted_class_label}')

plt.tight_layout()
plt.show()