# inventory_optimizer.py

import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from datetime import timedelta

class InventoryOptimizer:
    def __init__(self, data_path='data/', dashboard_data_path='dashboard_data/'):
        self.data_path = data_path
        self.dashboard_data_path = dashboard_data_path
        
        self.demand_df = None
        self.forecasts_df = None
        self.inventory_df = None
        self.products_df = None
        self.suppliers_df = None

        self._load_data()

    def _load_data(self):
        """Loads all necessary dataframes for inventory optimization."""
        try:
            self.demand_df = pd.read_csv(os.path.join(self.data_path, 'demand.csv'), parse_dates=['date'])
            self.demand_df.rename(columns={'date': 'ds', 'units_sold': 'y'}, inplace=True)
            self.demand_df = self.demand_df.dropna(subset=['ds', 'y'])
            self.demand_df['y'] = self.demand_df['y'].astype(int)

            self.forecasts_df = pd.read_csv(os.path.join(self.dashboard_data_path, 'fact_demand_forecasts.csv'), parse_dates=['forecast_date'])
            self.forecasts_df.rename(columns={'forecast_date': 'ds', 'forecasted_units': 'yhat'}, inplace=True)
            self.forecasts_df = self.forecasts_df.dropna(subset=['ds', 'yhat'])

            self.inventory_df = pd.read_csv(os.path.join(self.data_path, 'inventory.csv'), parse_dates=['inventory_date'])
            self.inventory_df.rename(columns={'quantity_on_hand': 'current_stock'}, inplace=True)
            self.inventory_df = self.inventory_df.dropna(subset=['current_stock'])
            self.inventory_df['current_stock'] = self.inventory_df['current_stock'].astype(int)

            self.products_df = pd.read_csv(os.path.join(self.data_path, 'dim_products.csv'))
            self.suppliers_df = pd.read_csv(os.path.join(self.data_path, 'dim_suppliers.csv'))

            print("InventoryOptimizer: All data loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading data for InventoryOptimizer: {e}. Ensure all data generation and forecasting steps are complete.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading in InventoryOptimizer: {e}")
            raise

    def calculate_inventory_metrics(self, service_level=0.95, default_lead_time_days=7, reorder_period_days=30):
        """
        Calculates Safety Stock, Reorder Point, and Reorder Quantity for each product-location.
        Optimized by pre-calculating aggregates.
        """
        if self.demand_df.empty or self.forecasts_df.empty or self.inventory_df.empty or self.suppliers_df.empty:
            print("Insufficient data to perform inventory optimization.")
            return pd.DataFrame()

        z_score = norm.ppf(service_level)

        
        # 1. Pre-calculate Average Daily Demand (ADD) and Standard Deviation of Daily Demand (SDD)
        #    for all product-location combinations from historical demand.
        historical_agg = self.demand_df.groupby(['product_id', 'location_id'])['y'].agg(
            avg_daily_demand='mean',
            std_dev_daily_demand='std'
        ).reset_index()
        
        # Fill NaN std_dev (e.g., if only one historical data point)
        historical_agg['std_dev_daily_demand'] = historical_agg['std_dev_daily_demand'].fillna(0)
        historical_agg['avg_daily_demand'] = historical_agg['avg_daily_demand'].fillna(0)


        # 2. Pre-calculate future forecasted demand sum for reorder quantity
        latest_historical_date = self.demand_df['ds'].max()
        forecast_period_start = latest_historical_date + timedelta(days=1)
        forecast_period_end = forecast_period_start + timedelta(days=reorder_period_days)

        future_forecasts_sum = self.forecasts_df[
            (self.forecasts_df['ds'] >= forecast_period_start) &
            (self.forecasts_df['ds'] < forecast_period_end)
        ].groupby(['product_id', 'location_id'])['yhat'].sum().reset_index()
        future_forecasts_sum.rename(columns={'yhat': 'forecasted_sum_reorder_period'}, inplace=True)


        # 3. Merge all necessary data into a single DataFrame to iterate or vectorize
        # Start with unique product-location combinations
        all_combinations = self.demand_df[['product_id', 'location_id']].drop_duplicates()

        # Merge historical aggregates (ADD, SDD)
        all_combinations = pd.merge(all_combinations, historical_agg, on=['product_id', 'location_id'], how='left')

        # Merge current stock
        all_combinations = pd.merge(all_combinations, self.inventory_df[['product_id', 'location_id', 'current_stock']], 
                                    on=['product_id', 'location_id'], how='left')
        all_combinations['current_stock'] = all_combinations['current_stock'].fillna(0).astype(int) # Fill missing with 0 and ensure int

        # Merge forecasted sum for ROQ
        all_combinations = pd.merge(all_combinations, future_forecasts_sum, on=['product_id', 'location_id'], how='left')
        all_combinations['forecasted_sum_reorder_period'] = all_combinations['forecasted_sum_reorder_period'].fillna(0)
        

        # 4. Calculate Inventory Metrics using vectorized operations or efficient iteration
        # Get average lead time (used globally for simplicity for now, could be per supplier/product later)
        avg_lead_time = self.suppliers_df['lead_time_days'].mean()
        if pd.isna(avg_lead_time):
            avg_lead_time = default_lead_time_days
        avg_lead_time = int(round(avg_lead_time)) # Ensure integer for calculations
        
        # Apply calculations using vectorized operations where possible
        all_combinations['avg_lead_time_days'] = avg_lead_time

        # Safety Stock (SS) = Z * SDD_LT ≈ Z * SDD * sqrt(Lead Time)
        all_combinations['safety_stock'] = (z_score * all_combinations['std_dev_daily_demand'] * np.sqrt(all_combinations['avg_lead_time_days'])).fillna(0).astype(int)
        all_combinations['safety_stock'] = all_combinations['safety_stock'].apply(lambda x: max(0, x))

        # Reorder Point (ROP) = (ADD * Lead Time) + Safety Stock
        all_combinations['reorder_point'] = (all_combinations['avg_daily_demand'] * all_combinations['avg_lead_time_days'] + 
                                             all_combinations['safety_stock']).fillna(0).astype(int)
        all_combinations['reorder_point'] = all_combinations['reorder_point'].apply(lambda x: max(0, x))

        # Reorder Quantity (ROQ) - using forecasted sum for reorder period
        all_combinations['reorder_quantity'] = all_combinations['forecasted_sum_reorder_period'].astype(int)
        all_combinations['reorder_quantity'] = all_combinations['reorder_quantity'].apply(lambda x: max(1, x) if x > 0 else int(max(1, all_combinations['avg_daily_demand'].mean() * reorder_period_days))) # Fallback if forecast is 0

        # 5. Determine Inventory Status (still best done with an apply or function for complex logic)
        def get_status(row):
            current = row['current_stock']
            reorder_p = row['reorder_point']
            safety_s = row['safety_stock']
            reorder_q = row['reorder_quantity']

            if current == 0:
                return "Out of Stock"
            elif current < safety_s:
                return "Critical (Below Safety Stock)"
            elif current <= reorder_p:
                return "Reorder Needed"
            elif current > (reorder_p + reorder_q * 1.5) and current > 0: # Arbitrary high threshold for overstock
                return "Potential Overstock"
            else:
                return "Optimal"

        all_combinations['inventory_status'] = all_combinations.apply(get_status, axis=1)

        # Select and reorder columns for final output
        final_df = all_combinations[[
            'product_id', 'location_id', 'current_stock', 'avg_daily_demand', 
            'avg_lead_time_days', 'safety_stock', 'reorder_point', 'reorder_quantity', 
            'inventory_status'
        ]]
        
        return final_df


# Example Usage (for testing the module directly)
if __name__ == "__main__":
    print("Running InventoryOptimizer example...")
        
    try:
        optimizer = InventoryOptimizer()
        inventory_insights_df = optimizer.calculate_inventory_metrics(
            service_level=0.95,
            default_lead_time_days=7,
            reorder_period_days=30
        )

        if not inventory_insights_df.empty:
            print("\n--- Sample Inventory Insights ---")
            print(inventory_insights_df.head())
            
            # Save results for dashboard
            output_file = os.path.join(optimizer.dashboard_data_path, 'inventory_insights.csv')
            inventory_insights_df.to_csv(output_file, index=False)
            print(f"\nInventory insights saved to {output_file}")
        else:
            print("No inventory insights generated. Check data or parameters.")
    except Exception as e:
        print(f"\nAn error occurred during inventory optimization: {e}")
