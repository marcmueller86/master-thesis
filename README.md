
# Project Overview

This repository contains a set of scripts designed to preprocess data, perform data analytics, and train models using neural ODEs (Ordinary Differential Equations).

## Project Structure

```
├── src/
│   ├── analysis_data.py
│   ├── customer_journey_analytics.py
│   ├── download_events_ode.py
│   ├── neural_ode.py
│   ├── smote_parallel.py
│   ├── split_neural_ode_data.py
│   └── split_target_ode.py
├── README.md
```

### Files in `src/` Directory

1. **analysis_data.py**  
   This script handles the analysis of data after model training or data preprocessing. It provides tools to visualize and statistically analyze datasets.

2. **customer_journey_analytics.py**  
   This script processes customer journey data, generating insights for understanding customer behavior. The analysis can be used for various applications such as targeted marketing and personalization.

3. **download_events_ode.py**  
   This script downloads and preprocesses event data for neural ODE training. It prepares time-series data, aligning it with the requirements for training neural ODE models.

4. **neural_ode.py**  
   This is the core script for training a Neural Ordinary Differential Equation (ODE) model. It utilizes the event data and models the time-series patterns to predict future events.

5. **smote_parallel.py**  
   This script implements SMOTE (Synthetic Minority Over-sampling Technique) in parallel to balance the dataset and reduce class imbalance issues. It uses multiprocessing to generate synthetic samples efficiently.

6. **split_neural_ode_data.py**  
   This script is responsible for splitting the dataset into training, validation, and testing sets for training neural ODE models. It ensures proper stratification and data integrity during the split process.

7. **split_target_ode.py**  
   This script prepares the target variables for neural ODE model training. It generates target labels that are compatible with ODE-based models by transforming the raw data into a suitable format.

## Setup Instructions

### Prerequisites

- Python 3.x
- Required libraries (install using `requirements.txt`):
  ```
  pip install -r requirements.txt
  ```

### Running the Scripts

1. **Data Download and Preprocessing**  
   Before training any models, use the following scripts to download and preprocess the data:
   ```bash
   python src/download_events_ode.py
   python src/split_neural_ode_data.py
   ```

2. **Customer Journey Analytics**  
   To run customer journey data analysis, use:
   ```bash
   python src/customer_journey_analytics.py
   ```

3. **SMOTE for Data Balancing**  
   To apply SMOTE with parallel processing:
   ```bash
   python src/smote_parallel.py
   ```

4. **Training the Neural ODE Model**  
   Once the data is ready, train the neural ODE model with:
   ```bash
   python src/neural_ode.py
   ```

5. **Data Analysis**  
   After training, you can analyze the results using:
   ```bash
   python src/analysis_data.py
   ```

## Notes

- Ensure that all dependencies are properly installed.
- Data must be preprocessed and split correctly before running the Neural ODE model.
- Use the SMOTE script to balance datasets with imbalanced classes for better model performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
