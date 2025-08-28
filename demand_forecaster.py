import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import warnings

# Suppress specific Prophet warnings if they become too verbose
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')


class DemandForecaster:
    def __init__(self, data_path='data/', output_path='dashboard_data/'):
        self.data_path = data_path
        self.output_path = output_path
        self.demand_df = None
        self._load_data()
        
        # Ensure output directory exists
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _load_data(self):
        """Loads historical demand data with robust NaN handling."""
        try:
            # Read CSV without initial date parsing to handle errors manually
            self.demand_df = pd.read_csv(f"{self.data_path}demand.csv")
            
            original_rows = len(self.demand_df)

            # Step 1: Robust Type Conversion and Initial NaN Handling
            # Convert 'date' column to datetime, coercing errors to NaT (Not a Time)
            self.demand_df['date'] = pd.to_datetime(self.demand_df['date'], errors='coerce')
            
            # Convert 'units_sold' to numeric, coercing errors to NaN, then fill NaN with 0
            # This ensures 'units_sold' is numeric and has no NaNs.
            self.demand_df['units_sold'] = pd.to_numeric(self.demand_df['units_sold'], errors='coerce').fillna(0)
            
            # Drop rows where 'date' became NaT during conversion
            # or if 'product_id'/'location_id' are missing (though they shouldn't be from generator)
            self.demand_df.dropna(subset=['date', 'product_id', 'location_id'], inplace=True)
            
            if len(self.demand_df) != original_rows:
                print(f"WARNING: Dropped {original_rows - len(self.demand_df)} rows with invalid dates or missing IDs during initial load cleanup.")

            # Final cast of units_sold to int (after all NaN handling)
            self.demand_df['units_sold'] = self.demand_df['units_sold'].astype(int)
            
            print("DemandForecaster: Historical demand data loaded and initially cleaned successfully.")
            print(f"Loaded {len(self.demand_df)} rows of demand data.")

        except FileNotFoundError as e:
            print(f"Error loading demand data: {e}. Make sure data_generator.py has been run and CSVs are in '{self.data_path}'")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            raise

    def _calculate_accuracy_metrics(self, actual, predicted):
        """Calculates MAE, RMSE, and MAPE."""
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Handle cases where actual demand might be zero to avoid division by zero in MAPE
        # For MAPE, we only consider points where actual > 0
        non_zero_actual_indices = actual > 0
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        if np.sum(non_zero_actual_indices) > 0:
            mape = np.mean(np.abs((actual[non_zero_actual_indices] - predicted[non_zero_actual_indices]) / actual[non_zero_actual_indices])) * 100
        else:
            mape = np.nan # Or 0 if there's no actual demand to compare against

        return mae, rmse, mape

    def forecast_all_demands(self, forecast_horizon_days=90, validation_days=30):
        """
        Iterates through all product-location combinations, forecasts demand,
        and calculates accuracy.
        """
        all_forecasts = []
        all_accuracy_metrics = []

        # Ensure the main demand_df is not empty after initial loading and cleaning
        if self.demand_df.empty:
            print("No valid demand data to process after initial loading. Exiting forecast.")
            return pd.DataFrame(), pd.DataFrame()

        unique_combinations = self.demand_df[['product_id', 'location_id']].drop_duplicates()
        print(f"Found {len(unique_combinations)} unique product-location combinations to forecast.")

        for index, row in unique_combinations.iterrows():
            product_id = row['product_id']
            location_id = row['location_id']

            print(f"\n--- Forecasting for Product: {product_id}, Location: {location_id} ---")

            # Step 2: Extract and Clean Time Series Data for Current Combination
            ts_data = self.demand_df[
                (self.demand_df['product_id'] == product_id) &
                (self.demand_df['location_id'] == location_id)
            ].copy() # Always work on a copy to avoid SettingWithCopyWarning

            # Rename columns to Prophet's expected 'ds' and 'y'
            ts_data = ts_data.rename(columns={'date': 'ds', 'units_sold': 'y'})
            
            # Aggressive NaN Removal for ds and y (Prophet's core columns)
            original_ts_len = len(ts_data)
            ts_data.dropna(subset=['ds', 'y'], inplace=True) # Ensure NO NaNs in ds or y
            
            if len(ts_data) != original_ts_len:
                print(f"  Dropped {original_ts_len - len(ts_data)} rows with NaNs in 'ds' or 'y' for P{product_id} L{location_id}.")

            # Ensure 'ds' is datetime and 'y' is float (Prophet's strict requirement)
            ts_data['ds'] = pd.to_datetime(ts_data['ds'], errors='coerce')
            ts_data['y'] = pd.to_numeric(ts_data['y'], errors='coerce')

            # Final dropna after type coercion, just in case something new turned into NaN
            original_ts_len = len(ts_data) # Re-check length
            ts_data.dropna(subset=['ds', 'y'], inplace=True)
            if len(ts_data) != original_ts_len:
                print(f"  Dropped {original_ts_len - len(ts_data)} rows after final type coercion cleanup for P{product_id} L{location_id}.")
            
            if ts_data.empty or len(ts_data) < 2: # Prophet needs at least 2 data points to fit
                print(f"  Insufficient valid historical data for P{product_id} L{location_id} ({len(ts_data)} points). Skipping forecast for this combination.")
                # Add NaN accuracy metrics if no forecast is possible
                all_accuracy_metrics.append({'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan})
                all_accuracy_metrics.append({'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan})
                all_accuracy_metrics.append({'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan})
                continue # Skip to the next product-location combination

            # Sort data by date for Prophet
            ts_data = ts_data.sort_values(by='ds').reset_index(drop=True)

            # Step 3: Validation Split
            # Ensure sufficient data points for validation period
            # Adjusted logic for validation_days_actual to ensure train_data is not too small
            # Need at least 2 points for validation if validation_days_actual > 0
            # And at least 2 points for training
            if len(ts_data) < validation_days + 4: # Small buffer for both train and validation
                validation_days_actual = max(0, len(ts_data) - 4) # Ensure train_data has at least 2-3 points
            else:
                validation_days_actual = validation_days

            train_data = ts_data[ts_data['ds'] < ts_data['ds'].max() - pd.Timedelta(days=validation_days_actual)].copy()
            validation_data = ts_data[ts_data['ds'] >= ts_data['ds'].max() - pd.Timedelta(days=validation_days_actual)].copy()

            # Step 4: Model Training & Accuracy Calculation
            current_accuracy_metrics = [] # To store metrics for this specific combination

            if train_data.empty or len(train_data) < 2:
                print(f"  Train data became too small ({len(train_data)} points) for P{product_id} L{location_id} after split. Using full ts_data for fit, skipping accuracy calculation.")
                # Fallback: Fit on full clean data if train_data is insufficient for a proper split
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                try:
                    model.fit(ts_data) # Fit on all available, cleaned data
                except Exception as e:
                    print(f"    Prophet fit failed for P{product_id} L{location_id} on full data: {e}. Skipping forecast.")
                    # Add NaN accuracy metrics if fit failed
                    current_accuracy_metrics.extend([
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan},
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan},
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan}
                    ])
                    all_accuracy_metrics.extend(current_accuracy_metrics)
                    continue # Skip to next combination

                # Add NaN accuracy metrics as no proper validation was done
                current_accuracy_metrics.extend([
                    {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan},
                    {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan},
                    {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan}
                ])
                
            else: # Sufficient train data for a proper validation split
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                try:
                    model.fit(train_data)
                except Exception as e:
                    print(f"    Prophet fit failed for P{product_id} L{location_id}: {e}. Skipping accuracy & forecast for this combination.")
                    # Add NaN accuracy metrics if fit failed
                    current_accuracy_metrics.extend([
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan},
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan},
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan}
                    ])
                    all_accuracy_metrics.extend(current_accuracy_metrics)
                    continue # Skip to next combination

                # Predict over the validation period using the trained model
                # Make future_dataframe *only for the validation period's dates*
                if not validation_data.empty: # Only predict if there's actual validation data
                    future_validation = model.make_future_dataframe(periods=validation_days_actual, include_history=False)
                    future_validation = future_validation[future_validation['ds'].isin(validation_data['ds'])].copy() # Filter to match validation dates
                    
                    if future_validation.empty or 'ds' not in future_validation.columns:
                        print(f"    Validation dataframe for prediction empty/invalid for P{product_id} L{location_id}. Skipping accuracy calculation.")
                        current_accuracy_metrics.extend([
                            {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan},
                            {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan},
                            {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan}
                        ])
                    else:
                        forecast_validation_df = model.predict(future_validation)
                        
                        # Align actuals and predictions carefully
                        actual_validation_aligned = validation_data.set_index('ds').reindex(forecast_validation_df['ds'])['y'].values
                        predicted_validation_aligned = forecast_validation_df['yhat'].values

                        # Calculate accuracy metrics, only if there are valid points for comparison
                        if np.sum(~np.isnan(actual_validation_aligned)) > 0 and np.sum(~np.isnan(predicted_validation_aligned)) > 0:
                            mae, rmse, mape = self._calculate_accuracy_metrics(
                                actual_validation_aligned[~np.isnan(actual_validation_aligned)], 
                                predicted_validation_aligned[~np.isnan(actual_validation_aligned)] # Only compare where actual is not NaN
                            )
                            current_accuracy_metrics.extend([
                                {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': mae},
                                {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': rmse},
                                {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': mape}
                            ])
                        else:
                            print(f"  Not enough valid data in validation set for P{product_id} L{location_id} to calculate accuracy after alignment. Skipping.")
                            current_accuracy_metrics.extend([
                                {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan},
                                {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan},
                                {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan}
                            ])
                else:
                    print(f"  No validation data available for P{product_id} L{location_id}. Skipping accuracy calculation.")
                    current_accuracy_metrics.extend([
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAE', 'Value': np.nan},
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'RMSE', 'Value': np.nan},
                        {'product_id': product_id, 'location_id': location_id, 'Metric': 'MAPE', 'Value': np.nan}
                    ])
            
            all_accuracy_metrics.extend(current_accuracy_metrics) # Add metrics for this combo

            # Step 5: Future Forecasting
            # Make future_dataframe from the *full* ts_data (train + validation) to forecast forward
            future_forecast = model.make_future_dataframe(periods=forecast_horizon_days, include_history=False)
            
            if future_forecast.empty or 'ds' not in future_forecast.columns:
                print(f"    Future forecast dataframe empty/invalid for P{product_id} L{location_id}. Skipping forecast output.")
                continue

            # This is where Prophet predicts, make sure future_forecast is clean
            forecast_result = model.predict(future_forecast)

            # Store results
            for_save = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy() # Include confidence intervals
            for_save.rename(columns={'ds': 'forecast_date', 'yhat': 'forecasted_units'}, inplace=True)
            for_save['product_id'] = product_id
            for_save['location_id'] = location_id
            
            # Ensure non-negative integers for forecasted units
            for_save['forecasted_units'] = for_save['forecasted_units'].apply(lambda x: max(0, int(np.round(x))))
            for_save['yhat_lower'] = for_save['yhat_lower'].apply(lambda x: max(0, int(np.round(x))))
            for_save['yhat_upper'] = for_save['yhat_upper'].apply(lambda x: max(0, int(np.round(x))))

            all_forecasts.append(for_save)
        
        # Concatenate all results into DataFrames
        final_forecasts_df = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
        final_accuracy_metrics_df = pd.DataFrame(all_accuracy_metrics)

        return final_forecasts_df, final_accuracy_metrics_df

    def save_forecasts_and_metrics(self, forecasts_df, accuracy_metrics_df):
        """Saves the generated dataframes to CSV."""
        if not forecasts_df.empty:
            forecasts_df.to_csv(f"{self.output_path}fact_demand_forecasts.csv", index=False)
            print(f"Forecasts saved to {self.output_path}fact_demand_forecasts.csv")
        else:
            print("No forecasts to save.")

        if not accuracy_metrics_df.empty:
            accuracy_metrics_df.to_csv(f"{self.output_path}metrics_forecast_accuracy.csv", index=False)
            print(f"Accuracy metrics saved to {self.output_path}metrics_forecast_accuracy.csv")
        else:
            print("No accuracy metrics to save.")

# Main execution block for demonstration
if __name__ == "__main__":
    forecaster = DemandForecaster()

    # Define forecast horizon and validation period
    FORECAST_HORIZON_DAYS = 90 # Predict 90 days into the future
    VALIDATION_DAYS = 30 # Use the last 30 days of historical data for accuracy validation

    forecasts, accuracy_metrics = forecaster.forecast_all_demands(
        forecast_horizon_days=FORECAST_HORIZON_DAYS,
        validation_days=VALIDATION_DAYS
    )

    forecaster.save_forecasts_and_metrics(forecasts, accuracy_metrics)

    print("\n--- Sample Forecasts ---")
    if not forecasts.empty:
        print(forecasts.head())
    else:
        print("No forecasts generated.")

    print("\n--- Sample Accuracy Metrics ---")
    if not accuracy_metrics.empty:
        print(accuracy_metrics.head())
    else:
        print("No accuracy metrics generated.")
