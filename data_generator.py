
# data_generator.py

import pandas as pd
import numpy as np
from faker import Faker
import random
import os
from datetime import timedelta

# Initialize Faker for realistic-looking data
fake = Faker()

# --- Configuration for Data Generation ---
NUM_PRODUCTS = 100
NUM_LOCATIONS = 20
NUM_SUPPLIERS = 10 # New: Number of suppliers
START_DATE = '2022-01-01'
END_DATE = '2024-12-31' # Historical data ends here
FORECAST_HORIZON_DAYS = 90 # Used to extend demand period for more realistic purchases
DATA_OUTPUT_DIR = 'data/'

# Ensure the output directory exists
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(DATA_OUTPUT_DIR)

print(f"Generating data to: {DATA_OUTPUT_DIR}")

# 1. Generate Dimension Tables

# dim_products.csv
print("Generating dim_products.csv...")
products_data = []
for i in range(1, NUM_PRODUCTS + 1):
    product_id = f"P{i:03d}"
    product_name = fake.word().capitalize() + " " + random.choice(["Gadget", "Tool", "Accessory", "Component", "Device", "Module"])
    category = random.choice(["Electronics", "Hardware", "Software", "Peripherals", "Networking", "Storage"])
    unit_cost = round(random.uniform(10, 500), 2)
    products_data.append([product_id, product_name, category, unit_cost])

dim_products_df = pd.DataFrame(products_data, columns=['product_id', 'product_name', 'category', 'unit_cost'])
dim_products_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'dim_products.csv'), index=False)
print("dim_products.csv generated.")


# dim_locations.csv
print("Generating dim_locations.csv...")
locations_data = []
for i in range(1, NUM_LOCATIONS + 1):
    location_id = f"L{i:02d}"
    city = fake.city()
    state = fake.state_abbr()
    region = random.choice(["North", "South", "East", "West", "Central"])
    locations_data.append([location_id, city, state, region])

dim_locations_df = pd.DataFrame(locations_data, columns=['location_id', 'city', 'state', 'region'])
dim_locations_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'dim_locations.csv'), index=False)
print("dim_locations.csv generated.")


# NEW: dim_suppliers.csv
print("Generating dim_suppliers.csv...")
suppliers_data = []
for i in range(1, NUM_SUPPLIERS + 1):
    supplier_id = f"S{i:02d}"
    supplier_name = fake.company() + " " + random.choice(["Supplies", "Solutions", "Global", "Corp"])
    contact_person = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace('\n', ', ')
    # Simulate average lead time, some suppliers are faster/slower
    lead_time_days = random.randint(3, 20) 
    suppliers_data.append([supplier_id, supplier_name, contact_person, phone, email, address, lead_time_days])

dim_suppliers_df = pd.DataFrame(suppliers_data, columns=['supplier_id', 'supplier_name', 'contact_person', 'phone', 'email', 'address', 'lead_time_days'])
dim_suppliers_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'dim_suppliers.csv'), index=False)
print("dim_suppliers.csv generated.")


# 2. Generate Fact Tables

# demand.csv (Historical Demand Data)
print("Generating demand.csv (historical fact table)...")
# Extend the date range for demand to cover the forecast horizon for more realistic purchase planning later
extended_end_date = pd.to_datetime(END_DATE) + timedelta(days=FORECAST_HORIZON_DAYS)
date_range = pd.date_range(start=START_DATE, end=extended_end_date, freq='D')
all_demand_data = []

# Simulate seasonal and trend components
base_demand = 50 # Average daily demand per product-location
daily_variation = 20 # Max random variation
seasonal_strength = 0.3 # How much seasonality impacts demand
trend_strength = 0.05 # How much trend impacts demand over time

for _, product_row in dim_products_df.iterrows():
    product_id = product_row['product_id']
    
    for _, location_row in dim_locations_df.iterrows():
        location_id = location_row['location_id']

        for i, current_date in enumerate(date_range):
            # Base demand
            demand = base_demand + random.uniform(-daily_variation, daily_variation)

            # Weekly seasonality (higher demand on weekends, lower on weekdays)
            if current_date.weekday() >= 4: # Friday (4), Saturday (5), Sunday (6)
                demand *= (1 + seasonal_strength * random.uniform(0.5, 1.5))
            else:
                demand *= (1 - seasonal_strength * random.uniform(0.1, 0.5))

            # Yearly seasonality (e.g., higher demand around holidays) - simplified
            if current_date.month in [11, 12, 1, 2]:
                demand *= (1 + seasonal_strength * random.uniform(0.3, 0.8))
            elif current_date.month in [6, 7]:
                 demand *= (1 + seasonal_strength * random.uniform(-0.2, 0.2))

            # Long-term trend
            trend_factor = (i / len(date_range)) * trend_strength * 100
            demand *= (1 + trend_factor / 100)

            units_sold = max(0, int(round(demand)))
            
            # Add some rare zeros for a more realistic distribution (e.g., no sales on a given day)
            if random.random() < 0.05:
                 units_sold = 0

            all_demand_data.append([
                current_date.strftime('%Y-%m-%d'),
                product_id,
                location_id,
                units_sold
            ])

demand_df = pd.DataFrame(all_demand_data, columns=['date', 'product_id', 'location_id', 'units_sold'])
demand_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'demand.csv'), index=False)
print("demand.csv generated.")


# NEW: inventory.csv (Snapshot of current inventory)
print("Generating inventory.csv (current stock levels)...")
inventory_data = []
for _, product_row in dim_products_df.iterrows():
    product_id = product_row['product_id']
    for _, location_row in dim_locations_df.iterrows():
        location_id = location_row['location_id']
        # Simulate initial stock levels based on average demand and some randomness
        initial_stock = max(0, int(round(random.gauss(base_demand * 10, base_demand * 3)))) # Gaussian distribution around 10 days of base demand
        inventory_data.append([
            pd.to_datetime(END_DATE).strftime('%Y-%m-%d'), # Current inventory as of the end of historical demand
            product_id,
            location_id,
            initial_stock
        ])

inventory_df = pd.DataFrame(inventory_data, columns=['inventory_date', 'product_id', 'location_id', 'quantity_on_hand'])
inventory_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'inventory.csv'), index=False)
print("inventory.csv generated.")


# NEW: fact_purchases.csv (Purchase Order History)
print("Generating fact_purchases.csv...")
purchases_data = []
purchase_id_counter = 1
purchase_start_date = pd.to_datetime(START_DATE) + timedelta(days=60) # Start purchases after some initial demand
purchase_end_date = pd.to_datetime(END_DATE)

for current_date in pd.date_range(start=purchase_start_date, end=purchase_end_date, freq='7D'): # Simulate orders roughly weekly
    # For each product-location, randomly decide if an order was placed
    for _, product_row in dim_products_df.iterrows():
        product_id = product_row['product_id']
        for _, location_row in dim_locations_df.iterrows():
            location_id = location_row['location_id']

            if random.random() < 0.3: # 30% chance of placing an order on this week for this product-location
                supplier = random.choice(dim_suppliers_df['supplier_id'].unique())
                
                # --- FIX Applied Here ---
                # Explicitly convert to int to avoid TypeError with timedelta
                supplier_lead_time = int(dim_suppliers_df[dim_suppliers_df['supplier_id'] == supplier]['lead_time_days'].iloc[0])
                # --- End Fix ---
                
                # Ordered quantity: base demand for a period, with randomness
                ordered_quantity = max(10, int(round(random.gauss(base_demand * 7, base_demand * 3)))) 
                unit_price = round(product_row['unit_cost'] * random.uniform(0.9, 1.2), 2) # Price can vary from cost
                
                delivery_date = current_date + timedelta(days=supplier_lead_time)
                
                # Simulate on-time delivery with some randomness (e.g., 90% on time)
                on_time = True
                if random.random() > 0.90: # 10% chance of late delivery
                    delivery_date += timedelta(days=random.randint(1, 7)) # 1-7 days late
                    on_time = False

                purchases_data.append([
                    f"PO{purchase_id_counter:06d}",
                    current_date.strftime('%Y-%m-%d'),
                    product_id,
                    location_id,
                    supplier,
                    ordered_quantity,
                    unit_price,
                    delivery_date.strftime('%Y-%m-%d'),
                    on_time
                ])
                purchase_id_counter += 1

fact_purchases_df = pd.DataFrame(purchases_data, columns=[
    'purchase_id', 'order_date', 'product_id', 'location_id', 
    'supplier_id', 'ordered_quantity', 'unit_price', 'delivery_date', 'on_time_delivery'
])
fact_purchases_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'fact_purchases.csv'), index=False)
print("fact_purchases.csv generated.")


print("\nAll data generation complete!")
print(f"Files saved in: {DATA_OUTPUT_DIR}")
print(f"  - dim_products.csv ({len(dim_products_df)} rows)")
print(f"  - dim_locations.csv ({len(dim_locations_df)} rows)")
print(f"  - dim_suppliers.csv ({len(dim_suppliers_df)} rows)")
print(f"  - demand.csv ({len(demand_df)} rows)")
print(f"  - inventory.csv ({len(inventory_df)} rows)")
print(f"  - fact_purchases.csv ({len(fact_purchases_df)} rows)")
