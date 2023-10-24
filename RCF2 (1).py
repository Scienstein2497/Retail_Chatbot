#!/usr/bin/env python
# coding: utf-8

# In[27]:


import streamlit as st
import sqlite3
import openai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import re
import codecs
import pandas as pd

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize OpenAI API key
openai.api_key = "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"

# Establish a connection to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('retail_data.db')
cur = conn.cursor()

# Define tables and columns based on the schema information

# Tables in the retail_data.db database
tables = [
    "BigSupplyCo_Orders",
    "BigSupplyCo_Products",
    "BigSupplyCo_Customers",
    "BigSupplyCo_Departments",
    "BigSupplyCo_Categories"
]

# Columns in each table
columns = {
    "BigSupplyCo_Orders": [
'order_id', 'order_item_cardprod_id', 'order_customer_id', 'order_department_id', 'market', 'order_city', 'order_country', 'order_region', 'order_state', 'order_status', 'order_zipcode', 'order_date__dateorders_', 'order_item_discount', 'order_item_discount_rate', 'order_item_id', 'order_item_quantity', 'sales', 'order_item_total', 'order_profit', 'type', 'days_for_shipping__real_', 'days_for_shipment__scheduled_', 'delivery_status', 'late_delivery_risk'       
    ],
    "BigSupplyCo_Products": ['product_card_id', 'product_category_id', 'product_description', 'product_image', 'product_name', 'product_price', 'product_status'

    ],
    "BigSupplyCo_Customers": [
'customer_id', 'customer_city', 'customer_country', 'customer_email', 'customer_fname', 'customer_lname', 'customer_password', 'customer_segment', 'customer_state', 'customer_street', 'customer_zipcode'
    ],
    "BigSupplyCo_Departments": [
        'department_id', 'department_name', 'latitude', 'longitude'
    ],
    "BigSupplyCo_Categories": [
        'category_id', 'category_name'
    ]
}


import openai

# Set your OpenAI API key
openai.api_key = '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'  # Replace 'your-api-key' with your actual API key

def generate_ai_response(prompt, tables, columns):
    try:
        # Use OpenAI GPT-3.5 to generate a natural language response
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150  # You can adjust the max_tokens based on the response length you need
        )

        # Get the generated response from OpenAI
        generated_response = response.choices[0].text.strip()

        # Tokenize user input
        tokens = word_tokenize(prompt.lower())
        # Check for keywords related to tables and columns
        selected_table = None
        selected_columns = []
        for table in tables:
            if table.lower() in tokens:
                selected_table = table
                selected_columns.extend(columns[table])

        if selected_table:
            # If a relevant table is found, construct the SQL query
            sql_query = f"SELECT {', '.join(selected_columns)} FROM {selected_table}"
            return generated_response, sql_query
        else:
            return generated_response, "No relevant table found in the input."
    except Exception as e:
        return str(e), str(e)

    
# Function to create table from CSV file
def create_table_from_csv(file_path, table_name, conn, encoding='utf-8'):
    with codecs.open(file_path, 'r', encoding=encoding, errors='replace') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read headers from CSV
        processed_headers = []
        for header in headers:
            # Replace spaces and special characters with underscores
            processed_header = re.sub(r'\W+', '_', header.lower())
            processed_headers.append(processed_header)
        headers_str = ', '.join(processed_headers)
        cur = conn.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({headers_str})")
        # Insert data from CSV into the table (assuming CSV format matches the table structure)
        for row in csvreader:
            cur.execute(f"INSERT INTO {table_name} VALUES ({','.join(['?']*len(row))})", row)

# File paths for the CSV data
departments_file = r'C:\Users\shara\Downloads\BigSupplyCo Data Files\BigSupplyCo Data Files\BigSupplyCo_Departments.csv'
categories_file = r'C:\Users\shara\Downloads\BigSupplyCo Data Files\BigSupplyCo Data Files\BigSupplyCo_Categories.csv'
customers_file = r'C:\Users\shara\Downloads\BigSupplyCo Data Files\BigSupplyCo Data Files\BigSupplyCo_Customers.csv'
products_file = r'C:\Users\shara\Downloads\BigSupplyCo Data Files\BigSupplyCo Data Files\BigSupplyCo_Products.csv'
orders_file = r'C:\Users\shara\Downloads\BigSupplyCo Data Files\BigSupplyCo Data Files\BigSupplyCo_Orders.csv'

# Create tables and import data from CSV files
create_table_from_csv(departments_file, 'BigSupplyCo_Departments', conn, encoding='ISO-8859-1')
create_table_from_csv(categories_file, 'BigSupplyCo_Categories', conn, encoding='ISO-8859-1')
create_table_from_csv(customers_file, 'BigSupplyCo_Customers', conn, encoding='ISO-8859-1')
create_table_from_csv(products_file, 'BigSupplyCo_Products', conn, encoding='ISO-8859-1')
create_table_from_csv(orders_file, 'BigSupplyCo_Orders', conn, encoding='ISO-8859-1')

# Commit changes and close the database connection
conn.commit()

# Streamlit UI
st.title("Retail Chatbot")
user_input = st.text_input("Enter your question:")
show_code = st.checkbox("Show Code")

# Main logic
if user_input:
    try:
        # Validate user input (ensure it is not empty or only whitespace)
        if user_input.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            # Generate AI response based on user input
            generated_response, sql_query = generate_ai_response(user_input, tables, columns)

            # Show generated AI response
            st.subheader("AI Response:")
            st.write(generated_response)

            # Correct the SQL code based on AI response
            corrected_sql_query = sql_query  # You can add logic here to modify the SQL query if needed

            # Show SQL code when the "Show Code" toggle button is enabled
            if show_code:
                st.subheader("Generated SQL Query:")
                st.code(corrected_sql_query, language='sql')  # Display corrected SQL code as a code block

                # Execute SQL query and display result
                cur.execute(corrected_sql_query)
                result = cur.fetchall()
                if result:
                    column_names = [description[0] for description in cur.description]
                    df = pd.DataFrame(result, columns=column_names)
                    st.subheader("Query Result:")
                    st.dataframe(df)  # Display DataFrame in Streamlit app
                else:
                    st.info("No results found.")
                
    except Exception as e:
        st.error(f"Error generating AI response or executing the query: {e}")

# Close Database Connection
conn.close()


# In[ ]:




