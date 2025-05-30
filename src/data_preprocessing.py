import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filepath):
    # Read the data without header and without dropping anything yet
    data = pd.read_csv(filepath, sep=r'\s+', header=None)

    # Remove any columns that are completely empty (extra spaces at the end of rows)
    data = data.dropna(axis=1, how='all')

    # Set proper column names (assumes 26 total)
    column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                   [f'sensor_measurement_{i}' for i in range(1, 22)]

    data.columns = column_names

    # Continue with rest of processing
    rul = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    data = data.merge(rul, on='unit_number', how='left')
    data['RUL'] = data['max_cycle'] - data['time_in_cycles']
    data.drop('max_cycle', axis=1, inplace=True)

    features = data.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
    labels = data['RUL']

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

