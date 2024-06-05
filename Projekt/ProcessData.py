import numpy, pandas, os
from Graphing import LABEL_CURRENT, LABEL_MAGNETIC_FIELD, LABEL_TIME, LABEL_BACKGROUND_TEMP, graph_data_against_time
from sklearn.linear_model import LinearRegression


def read_experiment_csv(experimentNumber: int, raw: bool) -> pandas.DataFrame:
    """
    Read a experiment data file,
    number of experiment passed as argument,
    raw flag indicates if raw or preprocessed data should be read,
    returns a pandas DataFrame.
    """
    filePath:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    if raw:
        filePath = os.path.join(filePath, (r"Data\Raw\Experiment"+str(experimentNumber)+".csv"))
    else:
        filePath = os.path.join(filePath, (r"Data\PreProcessed\Experiment"+str(experimentNumber)+".csv"))
    try:
        frame:pandas.DataFrame = pandas.read_csv(filePath)
        return frame
    except FileNotFoundError:
        print("read_experiment_csv failed, file not found, check if experiment data exists")
        raise   # Rethrow, this is not recoverable.
    
    

def save_processed_csv(dataFrame: pandas.DataFrame, experimentNumber: int) -> None:
    """
    Save a pandas DataFrame to a CSV file in the preprocessed data folder,
    number of experiment passed as argument,
    old data for the same experiment will be overwritten.
    """
    filePath:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    filePath = os.path.join(filePath, (r"Data\PreProcessed\Experiment"+str(experimentNumber)+".csv"))
    try:
        dataFrame.to_csv(filePath, index=False)
    except Exception:
        print("save_processed_csv failed.") # We can continue execution, but the data will be lost.

def remove_background_temperature(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    """
    Remove the background temperature from the experiment data,
    returns a new DataFrame with the background temperature removed.
    """
    dataFrame = dataFrame.drop(LABEL_BACKGROUND_TEMP, axis='columns')
    return dataFrame

def remove_outliers(dataFrame, x_col=LABEL_MAGNETIC_FIELD, y_col=LABEL_CURRENT, threshold=2.0):    #TODO
    # Fit a linear regression model
    X = dataFrame[[x_col]].values
    y = dataFrame[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the y values
    y_pred = model.predict(X)
    
    # Calculate the residuals
    residuals = y - y_pred
    
    # Calculate the standard deviation of the residuals
    std_residuals = numpy.std(residuals)
    
    # Determine the outlier threshold
    outlier_threshold = threshold * std_residuals
    
    # Identify outliers
    non_outliers = numpy.abs(residuals) <= outlier_threshold
    
    # Remove outliers from the dataframe and return
    dataFrame = dataFrame[non_outliers] 
    return dataFrame

def remove_invalid_magnetic_increases(dataFrame: pandas.DataFrame) -> pandas.DataFrame:
    """
    Returns a dataframe with invalid increases in magnetic field strength removed.
    Magnetic field should weaken with time as temeperature increases, an increase indicates invalid readings.
    Outliers should be removed before calling this function, they could introduce false minimas.
    """
    currentMinimumStrength = dataFrame.iloc[0][LABEL_MAGNETIC_FIELD]
    for index, row in dataFrame.iterrows():
        if row[LABEL_MAGNETIC_FIELD] < currentMinimumStrength:
            currentMinimumStrength = row[LABEL_MAGNETIC_FIELD]
        elif row[LABEL_MAGNETIC_FIELD] > currentMinimumStrength:
            dataFrame.drop(index, inplace=True)
    return dataFrame

def test() -> None:
    dataFrame = read_experiment_csv(1,True)
    #print(dataFrame)
    dataFrame = remove_background_temperature(dataFrame)
    graph_data_against_time(dataFrame, current=False, magneticField=True)
    #graph_data_against_time(remove_outliers(dataFrame), current=False, magneticField=True)
    remove_invalid_magnetic_increases(dataFrame)
    graph_data_against_time(dataFrame, current=False, magneticField=True)
test()
