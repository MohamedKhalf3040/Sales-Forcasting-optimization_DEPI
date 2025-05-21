import mysql.connector
import pandas as pd
import time

db_conn = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = '',
    database = 'Store_Sales_db'
)

cursor = db_conn.cursor()

training = pd.read_csv('./data/train.csv')

# casting
training['date'] = pd.to_datetime(training['date'])
training['family'] = training['family'].astype('category')
training.info()

for row in training.values:
    id = row[0]
    date = str(row[1]).replace(" 00:00:00", "")
    str_nb = row[2]
    fam = row[3]
    sales = row[4]
    onpr = row[5]
    sql = f"INSERT INTO training_5 (id, date, store_nbr, family, sales, onpromotion) VALUES ({id}, '{date}', {str_nb}, '{fam}', {sales}, {onpr})"
    try:
        cursor.execute(sql)
    except mysql.connector.Error as err:
        continue
    print(sql)
db_conn.commit()
print('Data inserted successfully')
cursor.close()
db_conn.close()