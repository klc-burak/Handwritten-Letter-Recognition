import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

# Veri setinin okunması
data = pd.read_csv("C:/Users/Burak/PycharmProjects/Biometric_Systems_Project/A_Z Handwritten Data.csv", low_memory=False)

# Veri setinin "features" ve "target" olarak ayrılması
features = data.iloc[:, 1:]
target = data.iloc[:, 0]

# Veri boyutlarını kontrol edilmesi
print("Data shape:",data.shape)
print("Features shape:", features.shape)
print("Target shape:", target.shape)

# Decision Tree algoritmasının tanımlanması
model_DT = DecisionTreeClassifier()

# En iyi özniteliğin seçimi için Decision Tree'nin kullanılması
selector = SelectFromModel(estimator=model_DT, max_features=10)
selected_features = selector.fit_transform(features, target)

# Training ve Test setlerinin belirlenmesi
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)

# Training set ile modelin eğitilmesi
model_DT.fit(X_train, y_train)

# Test seti üzerinden tahmin üretilmesi
predictions = model_DT.predict(X_test)

# Classification Accuracy değerinin hesaplanması ve yazdırılması
accuracy = accuracy_score(y_test, predictions)
print("Classification Accuracy:", accuracy)

# Confusion Matrix'in oluşturulması
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)