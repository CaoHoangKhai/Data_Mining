import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, precision_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pickle

# Đánh giá nghi thức K-fold
from sklearn.model_selection import cross_val_score

# Load data từ file Excel
data = pd.read_excel('Rice2024_v2.xlsx')

# Thông tin về dataset (kiểu dữ liệu, số lượng mẫu, giá trị null)
data.info()

print("========================")
# Số lượng mẫu và số lượng thuộc tính
print("Số lượng mẫu: ", len(data))
print("Số lượng thuộc tính: ", data.shape[1] - 1)  # Trừ đi 1 vì có cột 'Class'

# Tách dữ liệu thành X (các thuộc tính) và y (nhãn)
X = data.drop(columns=['Class'])
y = data['Class']

# Vẽ biểu đồ tần suất của biến mục tiêu
plt.hist(y, bins=10)
plt.title("Biểu đồ tần suất của biến mục tiêu")
plt.xlabel("Giá trị")
plt.ylabel("Tần suất")
# plt.show()

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo các mô hình
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier()

# Huấn luyện mô hình
knn_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Lưu các mô hình vào file pickle
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
pickle.dump(nb_model, open('nb_model.pkl', 'wb'))
pickle.dump(dt_model, open('dt_model.pkl', 'wb'))

# Chọn mô hình để dự đoán (thay thế model bằng một trong ba mô hình)
model = nb_model  # Sử dụng Gaussian Naive Bayes (thay đổi thành knn_model hoặc dt_model nếu cần)

# Dự đoán nhãn cho tập kiểm tra
y_pred = model.predict(X_test)

# In ra độ chính xác của mô hình với phương pháp kiểm tra hold-out
print("Độ chính xác của mô hình với phương pháp kiểm tra hold-out: %.3f" % accuracy_score(y_test, y_pred))

# Sử dụng nghi thức kiểm tra 20-fold (chia dữ liệu thành 20 phần)
nFold = 20
scores = cross_val_score(model, X, y, cv=nFold)

# In ra độ chính xác trung bình của mô hình với kiểm tra 20-fold
print("Độ chính xác của mô hình với nghi thức kiểm tra %d-fold: %.3f" % (nFold, np.mean(scores)))

# Tạo ma trận nhầm lẫn (confusion matrix)
cm = confusion_matrix(y_test, y_pred)

# Chuyển ma trận nhầm lẫn thành DataFrame để dễ dàng hiển thị
cm_df = pd.DataFrame(cm)

# Lấy nhãn lớp từ dữ liệu
labels = y.unique()

# Vẽ heatmap cho confusion matrix
plt.figure(figsize=(5.5, 4))
sns.heatmap(cm_df, fmt="d", annot=True, xticklabels=labels, yticklabels=labels)
plt.title(f'{model.__class__.__name__} \nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.show()

# Đánh giá mô hình qua các thước đo: recall, precision, f1-score và accuracy
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

# In ra các thước đo
print("Recall:", round(recall, 4))
print("Precision:", round(precision, 4))
print("F1 Score:", round(f1, 4))
print("Accuracy:", round(accuracy, 4))
