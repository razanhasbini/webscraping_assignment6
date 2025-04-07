from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("cleaned_ebay_deals.csv")
df.dropna(subset=['price', 'original_price', 'shipping', 'discount_percentage'], inplace=True)

def clean_shipping(value):
    if isinstance(value, str):
        if 'free' in value.lower():
            return 0.0
        try:
            return float(value.replace('$', '').strip())
        except:
            return np.nan
    return value

df['shipping'] = df['shipping'].apply(clean_shipping)
df.dropna(subset=['shipping'], inplace=True)

def categorize_discount(x):
    if x <= 10:
        return 'Low'
    elif x <= 30:
        return 'Medium'
    else:
        return 'High'

df['discount_bin'] = df['discount_percentage'].apply(categorize_discount)
print(df['discount_bin'].value_counts())
print("Shape after dropping missing values:", df.shape)

sns.histplot(df['discount_percentage'], kde=True)
plt.title("Distribution of Discount Percentage")
plt.xlabel('Discount Percentage')
plt.ylabel('Count')
plt.show()

min_size = df['discount_bin'].value_counts().min()
print("Smallest bin size is:", min_size)

df_balanced = df.groupby('discount_bin').apply(lambda x: x.sample(min_size)).reset_index(drop=True)
print(df_balanced['discount_bin'].value_counts())
print("Balanced shape:", df_balanced.shape)

df_balanced.drop(columns=['discount_bin'], inplace=True)

X = df_balanced[['price', 'original_price', 'shipping']]
y = df_balanced['discount_percentage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Discount %")
plt.ylabel("Predicted Discount %")
plt.title("Predicted vs Actual Discount Percentages")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

df_unseen = pd.read_csv('cleaned_ebay_deals.csv')
df_unseen = df_unseen.dropna(subset=['price', 'original_price', 'shipping'])

df_unseen['shipping'] = df_unseen['shipping'].apply(clean_shipping)
df_unseen.dropna(subset=['shipping'], inplace=True)

df_unseen = df_unseen.drop(columns=['discount_percentage'])
df_sample = df_unseen[['title', 'price', 'original_price', 'shipping']].sample(20).copy()
X_sample = df_sample[['price', 'original_price', 'shipping']]
df_sample['predicted_discount'] = model.predict(X_sample)

print(df_sample[['title', 'price', 'original_price', 'shipping', 'predicted_discount']])
