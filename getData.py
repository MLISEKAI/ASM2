import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sale.csv')

# Plotting Sales Over Time
df['SaleDate'] = pd.to_datetime(df['SaleDate'])
df.set_index('SaleDate', inplace=True)
sales_over_time = df['TotalPrice'].resample('ME').sum()

plt.figure(figsize=(10, 6))
plt.plot(sales_over_time, marker='o')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Product Sales Distribution
df = pd.read_csv('sale.csv')

product_sales = df.groupby('ProductID')['TotalPrice'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
product_sales.plot(kind='bar')
plt.title('Sales by Product')
plt.xlabel('Product ID')
plt.ylabel('Total Sales')
plt.show()

# Payment method distribution chart
df = pd.read_csv('sale.csv')

labels = ['Credit Card', 'Bank Transfer', 'PayPal']
sizes = [40, 35, 25]
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Payment Method Distribution')
plt.show()
