import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Check dataset labels
print("Dataset Distribution:")
print(df['v1'].value_counts())

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Custom Prediction
message = ["Congratulations! You won a free iPhone. Click now to claim your prize."]
message_tfidf = vectorizer.transform(message)

prediction = model.predict(message_tfidf)

print("\nCustom Message Prediction:")
if prediction[0] == 1:
    print("SPAM")
else:
    print("HAM")

# Visualization
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar')
plt.title("Spam vs Ham Messages")
plt.xlabel("Label (0 = Ham, 1 = Spam)")
plt.ylabel("Count")
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()