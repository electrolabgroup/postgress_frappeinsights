from datetime import datetime
import numpy as np

import psycopg2
import pandas as pd
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust level as needed

# Function to check if table exists
def table_exists(cursor, table_name):
    cursor.execute("""
    SELECT EXISTS (
       SELECT 1
       FROM   information_schema.tables 
       WHERE  table_schema = 'public'
       AND    table_name = %s
    );
    """, (table_name,))
    return cursor.fetchone()[0]

# Function to handle data retrieval and insertion for Opportunity
def fetch_and_insert_data_opportunity(url, params, headers, insert_query, cursor):
    limit_start = 0
    limit_page_length = 1000
    all_data = []
    while True:
        params['limit_start'] = limit_start
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                current_page_data = data['data']
                all_data.extend(current_page_data)
                if len(current_page_data) < limit_page_length:
                    break
                else:
                    limit_start += limit_page_length
            else:
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            break

    # Filter out records without name
    all_data_filtered = [record for record in all_data if 'name' in record and record['name'] is not None]

    # Normalize JSON data into a DataFrame
    opportunity = pd.json_normalize(all_data_filtered)

    # Insert data into PostgreSQL
    try:
        for row in opportunity.itertuples(index=False):
            cursor.execute(insert_query, (
                row.name,
                row.deal_pipeline,
                row.export_opportunity_amount,
                row.transaction_date,
                row.status,
                row.opportunity_amount,
                row.modified  # Last Updated On
            ))
        connection.commit()
        logging.info("Data insertion successful.")
    except psycopg2.Error as e:
        logging.error(f"Error inserting data into PostgreSQL: {e}")
        connection.rollback()  # Rollback changes in case of error

    return opportunity  # Return the DataFrame after insertion

# Define connection details for PostgreSQL
db_config = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'admin@123',
    'dbname': 'postgres'
}

# Connect to PostgreSQL
try:
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    # Define table name for Opportunity
    table_name_opportunity = 'Opportunity'

    # Check if Opportunity table exists, create if it doesn't
    if not table_exists(cursor, table_name_opportunity):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS Opportunity (
            name VARCHAR(255) PRIMARY KEY,
            deal_pipeline VARCHAR(255),
            export_opportunity_amount DECIMAL(15, 2),
            transaction_date DATE,
            status VARCHAR(50),
            opportunity_amount DECIMAL(15, 2),
            last_updated_on TIMESTAMP  -- New field for Last Updated On
        )
        """
        cursor.execute(create_table_query)
        logging.info("Opportunity table created.")

    # Define API endpoint and parameters for Opportunity
    base_url = 'https://erpv14.electrolabgroup.com/'
    endpoint = 'api/resource/Opportunity'
    url = base_url + endpoint
    params = {
        'fields': '["name","deal_pipeline","export_opportunity_amount","transaction_date","status","opportunity_amount","modified"]',
        'limit_page_length': 1000
    }
    headers = {
        'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
    }

    # Define INSERT query with conflict resolution for Opportunity
    insert_query = """
    INSERT INTO Opportunity (name, deal_pipeline, export_opportunity_amount, transaction_date, status, opportunity_amount, last_updated_on)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (name) DO UPDATE
    SET
        deal_pipeline = EXCLUDED.deal_pipeline,
        export_opportunity_amount = EXCLUDED.export_opportunity_amount,
        transaction_date = EXCLUDED.transaction_date,
        status = EXCLUDED.status,
        opportunity_amount = EXCLUDED.opportunity_amount,
        last_updated_on = EXCLUDED.last_updated_on  -- Update for Last Updated On
    """

    # Fetch data and insert into PostgreSQL for Opportunity, and capture DataFrame
    opportunity_df = fetch_and_insert_data_opportunity(url, params, headers, insert_query, cursor)

    # Now you can use opportunity_df for any further processing or analysis

except psycopg2.Error as e:
    logging.error(f"Error connecting to PostgreSQL: {e}")

finally:
    # Close cursor and connection
    if 'connection' in locals():
        cursor.close()
        connection.close()
        logging.info("PostgreSQL connection closed.")

import psycopg2
import pandas as pd
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust level as needed


# Function to check if table exists
def table_exists(cursor, table_name):
    cursor.execute("""
    SELECT EXISTS (
       SELECT 1
       FROM   information_schema.tables 
       WHERE  table_schema = 'public'
       AND    table_name = %s
    );
    """, (table_name,))
    return cursor.fetchone()[0]


# Function to handle data retrieval and insertion for Sales Order
def fetch_and_insert_data_sales_order(url, params, headers, insert_query, cursor):
    limit_start = 0
    limit_page_length = 1000
    all_data = []
    while True:
        params['limit_start'] = limit_start
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                current_page_data = data['data']
                all_data.extend(current_page_data)
                if len(current_page_data) < limit_page_length:
                    break
                else:
                    limit_start += limit_page_length
            else:
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            break

    # Filter out records without name
    all_data_filtered = [record for record in all_data if 'name' in record and record['name'] is not None]

    # Normalize JSON data into a DataFrame
    sales_order = pd.json_normalize(all_data_filtered)

    # Insert data into PostgreSQL
    try:
        for row in sales_order.itertuples(index=False):
            cursor.execute(insert_query, (
                row.name,
                row.transaction_date,
                row.net_total,
                row.naming_series
            ))
        connection.commit()
        logging.info("Data insertion successful.")
    except psycopg2.Error as e:
        logging.error(f"Error inserting data into PostgreSQL: {e}")
        connection.rollback()  # Rollback changes in case of error

    # Return the DataFrame
    return sales_order


# Define connection details for PostgreSQL
db_config = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'admin@123',
    'dbname': 'postgres'
}

# Connect to PostgreSQL
try:
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    # Define table name for Sales Order
    table_name_sales_order = 'SalesOrder'

    # Check if Sales Order table exists, create if it doesn't
    if not table_exists(cursor, table_name_sales_order):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS SalesOrder (
            name VARCHAR(255) PRIMARY KEY,
            transaction_date DATE,
            net_total DECIMAL(15, 2),
            naming_series VARCHAR(255)
        )
        """
        cursor.execute(create_table_query)
        logging.info("SalesOrder table created.")

    # Define API endpoint and parameters for Sales Order
    base_url = 'https://erpv14.electrolabgroup.com/'
    endpoint = 'api/resource/Sales Order'
    url = base_url + endpoint
    params = {
        'fields': '["name","transaction_date","net_total","naming_series"]',
        'limit_page_length': 1000
    }
    headers = {
        'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'  # Adjust authorization token
    }

    # Define INSERT query with conflict resolution for Sales Order
    insert_query = """
    INSERT INTO SalesOrder (name, transaction_date, net_total, naming_series)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (name) DO UPDATE
    SET
        transaction_date = EXCLUDED.transaction_date,
        net_total = EXCLUDED.net_total,
        naming_series = EXCLUDED.naming_series
    """

    # Fetch data and insert into PostgreSQL for Sales Order
    sales_order_df = fetch_and_insert_data_sales_order(url, params, headers, insert_query, cursor)
    logging.info(f"Data fetched and inserted into PostgreSQL. DataFrame shape: {sales_order_df.shape}")

except psycopg2.Error as e:
    logging.error(f"Error connecting to PostgreSQL: {e}")

finally:
    # Close cursor and connection
    if 'connection' in locals():
        cursor.close()
        connection.close()
        logging.info("PostgreSQL connection closed.")



# Convert 'transaction_date' and 'modified' to datetime
opportunity_df['transaction_date'] = pd.to_datetime(opportunity_df['transaction_date'])
opportunity_df['modified'] = pd.to_datetime(opportunity_df['modified'])

# Get current date
current_date = datetime.now()

# Filter conditions
transaction_date_condition = opportunity_df['transaction_date'].dt.month != current_date.month
status_condition = opportunity_df['status'].isin(['Closed', 'Converted', 'Order Won', 'Lost', 'Order Lost'])
deal_pipeline_condition = opportunity_df['deal_pipeline'].str.contains('export', case=False)
modified_condition = (opportunity_df['modified'].dt.year == current_date.year) & (opportunity_df['modified'].dt.month == current_date.month)

# Apply filters
opportunity_df_filtered = opportunity_df[
    transaction_date_condition &
    status_condition &
    deal_pipeline_condition &
    modified_condition
]



# Convert 'transaction_date' and 'modified' to datetime
opportunity_df_filtered['transaction_date'] = pd.to_datetime(opportunity_df_filtered['transaction_date'])
opportunity_df_filtered['modified'] = pd.to_datetime(opportunity_df_filtered['modified'])

# Extract month and year from 'transaction_date'
opportunity_df_filtered['transaction_month'] = opportunity_df_filtered['transaction_date'].dt.to_period('M')

# Group by 'deal_pipeline' and 'transaction_month', and sum 'export_opportunity_amount'
opportunity = opportunity_df_filtered.groupby(['deal_pipeline', 'transaction_month'])['export_opportunity_amount'].sum().reset_index()


# Replace the values in 'deal_pipeline' column with 'Export'
opportunity['deal_pipeline'] = 'Export'

# Group by 'transaction_month' and 'deal_pipeline', and sum 'export_opportunity_amount'
opp_export_df = opportunity.groupby(['deal_pipeline'])['export_opportunity_amount'].sum().reset_index()



# Get the current month in the format 'YYYY-MM'
current_month = datetime.now().strftime('%Y-%m')

# Add the current month as a new column
opp_export_df['month'] = current_month



# Get the current date
current_date = datetime.now()

# Filter out current month
carry_export = opportunity_df[opportunity_df['transaction_date'].dt.month != current_date.month]



# Exclude specific statuses
excluded_statuses = ['Closed', 'Converted', 'Order Won', 'Lost', 'Order Lost']
carry_export = carry_export[~carry_export['status'].isin(excluded_statuses)]



# Filter deal_pipeline to include 'export' (case insensitive)
carry_export = carry_export.dropna(subset=['deal_pipeline'])
deal_pipeline_condition = carry_export['deal_pipeline'].str.contains('export', case=False)
carry_export = carry_export[deal_pipeline_condition]



# Extract month and year from 'transaction_date'
carry_export['transaction_month'] = carry_export['transaction_date'].dt.to_period('M')

# Group by 'deal_pipeline' and 'transaction_month', and sum 'export_opportunity_amount'
carry_export = carry_export.groupby(['deal_pipeline', 'transaction_month'])['export_opportunity_amount'].sum().reset_index()



# Replace the values in 'deal_pipeline' column with 'Export'
carry_export['deal_pipeline'] = 'Export'

# Group by 'transaction_month' and 'deal_pipeline', and sum 'export_opportunity_amount'
carry_export = carry_export.groupby(['deal_pipeline'])['export_opportunity_amount'].sum().reset_index()



current_month = datetime.now().strftime('%Y-%m')

# Add the current month as a new column
carry_export['month'] = current_month



# Convert 'transaction_date' and 'modified' to datetime
opportunity_df['transaction_date'] = pd.to_datetime(opportunity_df['transaction_date'])
opportunity_df['modified'] = pd.to_datetime(opportunity_df['modified'])

# Get current date
current_date = datetime.now()

# Filter conditions
transaction_date_condition = opportunity_df['transaction_date'].dt.month != current_date.month
status_condition = opportunity_df['status'].isin(['Closed', 'Converted', 'Order Won', 'Lost', 'Order Lost'])

modified_condition = (opportunity_df['modified'].dt.year == current_date.year) & (opportunity_df['modified'].dt.month == current_date.month)

# Apply filters
opportunity_df_filtered = opportunity_df[
    transaction_date_condition &
    status_condition &
    modified_condition
]


# Convert 'transaction_date' and 'modified' to datetime
opportunity_df_filtered['transaction_date'] = pd.to_datetime(opportunity_df_filtered['transaction_date'])
opportunity_df_filtered['modified'] = pd.to_datetime(opportunity_df_filtered['modified'])

# Extract month and year from 'transaction_date'
opportunity_df_filtered['transaction_month'] = opportunity_df_filtered['transaction_date'].dt.to_period('M')

# Group by 'deal_pipeline' and 'transaction_month', and sum 'export_opportunity_amount'
opportunity = opportunity_df_filtered.groupby(['deal_pipeline', 'transaction_month'])['opportunity_amount'].sum().reset_index()




# Group by 'transaction_month' and 'deal_pipeline', and sum 'export_opportunity_amount'
opp_dom_df = opportunity.groupby(['deal_pipeline'])['opportunity_amount'].sum().reset_index()



# Calculate the sum of opportunity amounts for 'Export Machine' and 'Export Spares'
export_sum = opp_dom_df.loc[opp_dom_df['deal_pipeline'].isin(['Export Machine', 'Export Spares']), 'opportunity_amount'].sum()

# Replace 'Export Machine' and 'Export Spares' with 'Export'
opp_dom_df.loc[opp_dom_df['deal_pipeline'].isin(['Export Machine', 'Export Spares']), 'deal_pipeline'] = 'Export'

# Group by 'deal_pipeline' and sum the opportunity amounts
opp_dom_df = opp_dom_df.groupby('deal_pipeline', as_index=False).sum()



# Get the current date
current_date = datetime.now()

# Filter out current month
carry_domestic = opportunity_df[opportunity_df['transaction_date'].dt.month != current_date.month]



# Exclude specific statuses
excluded_statuses = ['Closed', 'Converted', 'Order Won', 'Lost', 'Order Lost']
carry_domestic = carry_domestic[~carry_domestic['status'].isin(excluded_statuses)]


# Filter deal_pipeline to include 'export' (case insensitive)
carry_domestic = carry_domestic.dropna(subset=['deal_pipeline'])
carry_domestic.head()



# Group by 'transaction_month' and 'deal_pipeline', and sum 'export_opportunity_amount'
carry_domestic = carry_domestic.groupby(['deal_pipeline'])['opportunity_amount'].sum().reset_index()
carry_domestic.head()




# Calculate the sum of opportunity amounts for 'Export Machine' and 'Export Spares'
export_sum = carry_domestic.loc[carry_domestic['deal_pipeline'].isin(['Export Machine', 'Export Spares']), 'opportunity_amount'].sum()

# Replace 'Export Machine' and 'Export Spares' with 'Export'
carry_domestic.loc[carry_domestic['deal_pipeline'].isin(['Export Machine', 'Export Spares']), 'deal_pipeline'] = 'Export'

# Group by 'deal_pipeline' and sum the opportunity amounts
carry_domestic = carry_domestic.groupby('deal_pipeline', as_index=False).sum()
carry_domestic.head()




current_month = datetime.now().strftime('%Y-%m')

# Add the current month as a new column
carry_domestic['month'] = current_month
carry_domestic.head()




current_month = datetime.now().strftime('%Y-%m')

# Add the current month as a new column
opp_dom_df['month'] = current_month
opp_dom_df.head()




carry_export.rename(columns = {'export_opportunity_amount':'opportunity_amount'}, inplace = True)
carry_export.head()




opp_export_df.rename(columns = {'export_opportunity_amount':'opportunity_amount'}, inplace = True)
opp_export_df.head()



# Concatenate all dataframes
all_data = pd.concat([carry_domestic, opp_dom_df, carry_export, opp_export_df])

# Group by 'deal_pipeline' and sum 'opportunity_amount'
opp_final = all_data.groupby('deal_pipeline', as_index=False)['opportunity_amount'].sum()
opp_final.head()




# List of deal pipelines to be grouped under 'Machine'
machine_pipelines = ['Machine', 'Peristaltic Pump', 'Star Series Pump', 'Formulation R & D',
                     'Trial', 'Biowise', 'Bioreactor', 'Aquaflux', 'Product Specialist']

# List of deal pipelines to be grouped under 'GastroSimPlus'
gastrosimplus_pipelines = ['Gastro', 'SimPlus']

# Create a new column 'group' to classify deal pipelines
def classify_pipeline(pipeline):
    if pipeline in machine_pipelines:
        return 'Machine'
    elif pipeline in gastrosimplus_pipelines:
        return 'Gastro + Simplus'
    else:
        return pipeline

opp_final['group'] = opp_final['deal_pipeline'].apply(classify_pipeline)

# Group by the new column 'group' and sum the 'opportunity_amount'
grouped_opp_final = opp_final.groupby('group', as_index=False)['opportunity_amount'].sum()




# Filter to keep only the specified groups
desired_groups = ['Machine', 'Spares', 'Service', 'Export', 'Gastro + Simplus']
filtered_grouped_opp_final = grouped_opp_final[grouped_opp_final['group'].isin(desired_groups)]
filtered_grouped_opp_final.head()



# Rename 'Service' to 'Assurance'
filtered_grouped_opp_final['group'] = filtered_grouped_opp_final['group'].replace('Service', 'Assurance')





current_month = datetime.now().strftime('%Y-%m')

# Add the current month as a new column
filtered_grouped_opp_final['year_month'] = current_month



# Mapping of naming_series to new values
naming_series_mapping = {
    '2324SODM.####': 'Machine',
    '2324SOEXP.####': 'Export',
    '2324SODS.####': 'Spares',
    '2324SOSA.####': 'Assurance',
    '2324SOSP.####': 'Gastro + Simplus',
    '2425SODM.####': 'Machine',
    '2425SOEXP.####': 'Export',
    '2425SODS.####': 'Spares',
    '2425SOSA.####': 'Assurance',
    '2425SOSP.####': 'Gastro + Simplus'
}

# Filter the dataframe to keep only the rows with the specified naming_series
filtered_df = sales_order_df[sales_order_df['naming_series'].isin(naming_series_mapping.keys())]

# Replace the naming_series values according to the mapping
filtered_df['naming_series'] = filtered_df['naming_series'].replace(naming_series_mapping)

# Replace the name values accordingly (example logic, you can adjust based on your specific rules)
filtered_df['name'] = filtered_df['name'].apply(lambda x: x.replace('SODM', 'Machine')
                                                            .replace('SOEXP', 'Export')
                                                            .replace('SODS', 'Spares')
                                                            .replace('SOSA', 'Assurance')
                                                            .replace('SOSP', 'Gastro + Simplus'))



# Convert 'transaction_date' to datetime
filtered_df['transaction_date'] = pd.to_datetime(filtered_df['transaction_date'])

# Extract year and month
filtered_df['year_month'] = filtered_df['transaction_date'].dt.to_period('M')

# Group by 'year_month' and 'naming_series' and sum 'net_total'
so_df = filtered_df.groupby(['year_month', 'naming_series'])['net_total'].sum().reset_index()



so_df.rename(columns = {'naming_series':'group'}, inplace = True)




# Convert 'year_month' in filtered_grouped_opp_final from object to period[M]
filtered_grouped_opp_final['year_month'] = pd.to_datetime(filtered_grouped_opp_final['year_month']).dt.to_period('M')

print(so_df['year_month'].dtype)
print(filtered_grouped_opp_final['year_month'].dtype)



# Perform the inner merge
merged_df = pd.merge(so_df, filtered_grouped_opp_final, on=['year_month', 'group'], how='inner')




# Calculate the conversion ratio
merged_df['conversion_ratio'] = merged_df['net_total'] / merged_df['opportunity_amount']




print(merged_df['year_month'].dtype)



# Convert the ratio to a percentage
merged_df['conversion_rate'] = np.round(merged_df['conversion_ratio'] * 100, 2)




# Convert 'year_month' to string before converting to datetime
merged_df['year_month'] = merged_df['year_month'].astype(str)

# Get the current date and time
current_datetime = datetime.now()

# Convert 'year_month' to datetime with current date and time
merged_df['year_month'] = pd.to_datetime(merged_df['year_month'] + '-' + current_datetime.strftime('%d %H:%M:%S'))

import psycopg2
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust level as needed


def prepare_dataframe(df):
    # Convert 'year_month' to string if it is in Period format
    if pd.api.types.is_period_dtype(df['year_month']):
        df['year_month'] = df['year_month'].astype(str)

    return df


# Define connection details for PostgreSQL
db_config = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'admin@123',
    'dbname': 'postgres'
}


# Function to check if table exists
def table_exists(cursor, table_name):
    cursor.execute("""
    SELECT EXISTS (
       SELECT 1
       FROM   information_schema.tables 
       WHERE  table_schema = 'public'
       AND    table_name = %s
    );
    """, (table_name,))
    return cursor.fetchone()[0]


# Function to create table
def create_table(cursor):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS SalesMetrics (
        year_month TIMESTAMP,
        "group" VARCHAR(255),
        net_total DECIMAL(15, 2),
        opportunity_amount DECIMAL(15, 2),
        conversion_ratio DECIMAL(10, 6),
        conversion_rate DECIMAL(5, 2),
        PRIMARY KEY (year_month, "group")
    )
    """
    cursor.execute(create_table_query)
    logging.info("SalesMetrics table created.")


# Function to check if current year_month exists
def current_month_data_exists(cursor, current_year_month):
    cursor.execute("""
    SELECT EXISTS (
        SELECT 1
        FROM SalesMetrics
        WHERE year_month >= %s AND year_month < %s
    );
    """, (current_year_month, current_year_month + pd.DateOffset(months=1)))
    return cursor.fetchone()[0]


# Function to delete data for the current month
def delete_current_month_data(cursor, current_year_month):
    cursor.execute("""
    DELETE FROM SalesMetrics
    WHERE year_month >= %s AND year_month < %s
    """, (current_year_month, current_year_month + pd.DateOffset(months=1)))
    connection.commit()
    logging.info("Existing data for the current month deleted.")


# Function to insert or update data
def insert_or_update_data(df, cursor):
    df = prepare_dataframe(df)  # Ensure data types are compatible

    insert_query = """
    INSERT INTO SalesMetrics (year_month, "group", net_total, opportunity_amount, conversion_ratio, conversion_rate)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (year_month, "group") DO UPDATE
    SET
        net_total = EXCLUDED.net_total,
        opportunity_amount = EXCLUDED.opportunity_amount,
        conversion_ratio = EXCLUDED.conversion_ratio,
        conversion_rate = EXCLUDED.conversion_rate
    """

    try:
        for row in df.itertuples(index=False):
            cursor.execute(insert_query, (
                row.year_month,
                row.group,
                row.net_total,
                row.opportunity_amount,
                row.conversion_ratio,
                row.conversion_rate
            ))
        connection.commit()
        logging.info("Data insertion/update successful.")
    except psycopg2.Error as e:
        logging.error(f"Error inserting/updating data into PostgreSQL: {e}")
        connection.rollback()  # Rollback changes in case of error


# Get current year and month
current_datetime = datetime.now()
current_year_month = pd.Timestamp(current_datetime.strftime('%Y-%m-01'))

# Convert 'year_month' to string before converting to datetime
merged_df['year_month'] = merged_df['year_month'].astype(str)

# Form valid datetime strings and convert to datetime objects
merged_df['year_month'] = pd.to_datetime(merged_df['year_month'] + '-01 ' + current_datetime.strftime('%H:%M:%S'))

# Connect to PostgreSQL
try:
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    # Define table name
    table_name = 'SalesMetrics'

    # Check if table exists, create if it doesn't
    if not table_exists(cursor, table_name):
        create_table(cursor)

    # Check if current month data exists in the table
    if current_month_data_exists(cursor, current_year_month):
        # Delete the data for the current month
        delete_current_month_data(cursor, current_year_month)

    # Insert the new data
    insert_or_update_data(merged_df, cursor)

except psycopg2.Error as e:
    logging.error(f"Error connecting to PostgreSQL: {e}")

finally:
    # Close cursor and connection
    if 'connection' in locals():
        cursor.close()
        connection.close()
        logging.info("PostgreSQL connection closed.")







