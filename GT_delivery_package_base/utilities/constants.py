import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, 'Data')
model_dir = os.path.join(project_dir,'models')
incremental_data_dir = os.path.join(data_dir, 'incremental_data')

base_file = "scantrack_base_monthly_v2.csv"
selected_features_filename = "selected_features.csv"
template_filename = "simulated_data.xlsx"
sales_pred_filename = "sales_prediction.csv"
feature_importance_filename = "feature_importance_data.csv"
ppu_tdp_pred_filename = "ppu_tdp_prediction.csv"
scores_filename = "model_score_tracker.csv"

sales_error_file = "sales_error_tracker.csv"
som_error_file = "som_error_tracker.csv"
ppu_error_file = "ppu_error_tracker.csv"
tdp_error_file = "tdp_error_tracker.csv"
units_error_file = "units_error_tracker.csv"
sales_req_file = "error_tracker_per_month.csv"
ppu_tdp_req_file = "ppu_tdp_error_tracker_per_month.csv"

realistic_simulation_file = 'realistic_simulation.csv'

future_months = 18
test_months=3

# sort column order
sortList = ['PEPSICO PPU', 'PEPSICO TDP', 'DIANA PPU', 'DIANA TDP', 'BIMBO PPU', 'BIMBO TDP', 
            'KELLOGGS PPU', 'KELLOGGS TDP', 'SEÑORIAL PPU', 'SEÑORIAL TDP', 'DINANT PPU', 'DINANT TDP',
            'OTHERS PPU', 'OTHERS TDP']

month_dict = {'ENERO': 'JANUARY', 'FEBRERO': 'FEBRUARY', 'MARZO': 'MARCH',
              'ABRIL': 'APRIL', 'MAYO': 'MAY', 'JUNIO':'JUNE', 'JULIO': 'JULY',
              'AGOSTO': 'AUGUST', 'SEPTIEMBRE': 'SEPTEMBER', 'OCTUBRE': 'OCTOBER',
              'NOVIEMBRE': 'NOVEMBER', 'DICIEMBRE': 'DECEMBER'}

short_month_dict = {'ENE': 'JANUARY', 'FEB': 'FEBRUARY', 'MAR': 'MARCH',
              'ABR': 'APRIL', 'MAY': 'MAY', 'JUN':'JUNE', 'JUL': 'JULY',
              'AGO': 'AUGUST', 'SEP': 'SEPTEMBER', 'OCT': 'OCTOBER',
              'NOV': 'NOVEMBER', 'DIC': 'DECEMBER'}

regression_features =  ['BIMBO PPU', 'DIANA PPU', 'KELLOGGS PPU', 'OTHERS PPU', 'PEPSICO PPU', 'SEÑORIAL PPU', 'DINANT PPU', 
                        'BIMBO TDP', 'DIANA TDP', 'KELLOGGS TDP', 'OTHERS TDP', 'PEPSICO TDP', 'SEÑORIAL TDP', 'DINANT TDP', 
                        'BIMBO PPU Ratio', 'DIANA PPU Ratio', 'KELLOGGS PPU Ratio',  'OTHERS PPU Ratio', 
                        'PEPSICO PPU Ratio', 'SEÑORIAL PPU Ratio', 'DINANT PPU Ratio', 'BIMBO TDP Ratio', 
                        'DIANA TDP Ratio', 'KELLOGGS TDP Ratio', 'OTHERS TDP Ratio', 'PEPSICO TDP Ratio',
                        'SEÑORIAL TDP Ratio', 'DINANT TDP Ratio']

seasonality_cols = [
                     'monthly_index', 
                     'quarterly_index'
                    ]

# set bounds
# For avg_price, tdp and generated features
max_feat_bounds = [-0.001, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999,
                   999999999, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001 
                  ]
min_feat_bounds = [-999999999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                   0.001, -999999999, -999999999, -999999999, -999999999, -999999999, -999999999 
                  ]

# for seasonality
max_month_bounds = [999999999] * 12
min_month_bounds = [-999999999] * 12

# for monthly and quarterly indices
seasonality_min_bounds = [-999999999,
                          -999999999
                         ]
seasonality_max_bounds = [999999999, 
                          999999999
                         ]

# parameters to tune
params = {
    'intercept': [(-999999999, 999999999), (0.001, 999999999), (-999999999, -0.001)]
}