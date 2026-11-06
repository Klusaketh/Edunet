# EV Adoption Forecasting

Forecasts electric vehicle adoption by US county for the next three years, to help plan charging infrastructure. Built during the AICTE internship (Cycle 2) with Edunet Foundation.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Dataset

Electric Vehicle Population Size History by County - 20,819 monthly records with BEV, PHEV, EV total and non-EV counts per county.

## Approach

1. **Cleaning** - parse dates, fill missing county/state values, cap outliers in EV percentage using IQR bounds
2. **EDA** - counties with the most and fewest EVs, BEV vs PHEV totals
3. **Feature engineering** - months since start, encoded county, 1-3 month lags, rolling mean, percent change and a 6-month growth slope of cumulative EVs
4. **Model** - Random Forest Regressor tuned with `RandomizedSearchCV` (3-fold, 30 candidates), time-ordered train-test split
5. **Evaluation** - MAE, RMSE, R², actual vs predicted plot and feature importance
6. **Forecast** - 36-month cumulative EV forecast for a single county and the top 5 counties (Santa Clara, Fairfax, Orange, Honolulu, Los Angeles)
7. **Deployment prep** - trained model saved with `joblib` and reloaded for prediction

## Files

All files are in [`EV_Charging_pred-main/`](EV_Charging_pred-main):

| File | Description |
|---|---|
| `main.ipynb` | Full notebook with outputs |
| `Electric_Vehicle_Population_Size_History_By_County_.csv` | Raw dataset |
| `preprocessed_ev_data.csv` | Engineered features |
| `forecasting_ev_model.pkl` | Trained model |

## Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
cd EV_Charging_pred-main
jupyter notebook main.ipynb
```

## Author

**Paramkusam Saketha Sreevatsa** - [GitHub](https://github.com/Klusaketh) · [LinkedIn](https://www.linkedin.com/in/paramkusamsakethasreevatsa)
