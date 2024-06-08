import os
import numpy
import pandas
from ydata_profiling import ProfileReport   # Newer version of pandas-profiling
from Graphing import LABEL_CURRENT, LABEL_MAGNETIC_FIELD, LABEL_BACKGROUND_TEMP, graph_polynomial_fit


def read_experiment_csv(experiment_number: int, raw: bool) -> pandas.DataFrame:
    """
    Read a experiment data file,
    number of experiment passed as argument,
    raw flag indicates if raw or preprocessed data should be read,
    returns a pandas DataFrame.
    """
    file_path:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    if raw:
        file_path = os.path.join(file_path, (r"Data\Raw\Experiment"+str(experiment_number)+".csv"))
    else:
        file_path = os.path.join(file_path, (r"Data\PreProcessed\Experiment"+str(experiment_number)+".csv"))
    try:
        frame:pandas.DataFrame = pandas.read_csv(file_path)
        return frame
    except FileNotFoundError:
        print("read_experiment_csv failed, file not found, check if experiment data exists")
        raise   # Rethrow, this is not recoverable.


def save_processed_csv(data_frame: pandas.DataFrame, experiment_number: int) -> None:
    """
    Save a pandas DataFrame to a CSV file in the preprocessed data folder,
    number of experiment passed as argument,
    old data for the same experiment will be overwritten.
    """
    file_path:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    file_path = os.path.join(file_path, (r"Data\PreProcessed\Experiment"+str(experiment_number)+".csv"))
    try:
        data_frame.to_csv(file_path, index=False)
    except Exception:
        print("save_processed_csv failed.") # We can continue execution, but the data will be lost.


def remove_background_temperature(data_frame: pandas.DataFrame) -> None:
    """
    Remove the background temperature from the experiment data,
    returns a new DataFrame with the background temperature removed.
    """
    data_frame = data_frame.drop(LABEL_BACKGROUND_TEMP, axis='columns')


def remove_invalid_magnetic_increases(data_frame: pandas.DataFrame) -> None:
    """
    Removes invalid increases in magnetic field strength from dataframe.
    Magnetic field should weaken with time as temeperature increases, an increase indicates invalid readings.
    Outliers should be removed before calling this function, they could introduce false minimas.
    """
    current_minimum_strength = data_frame.iloc[0][LABEL_MAGNETIC_FIELD]
    for index, row in data_frame.iterrows():
        if row[LABEL_MAGNETIC_FIELD] < current_minimum_strength:
            current_minimum_strength = row[LABEL_MAGNETIC_FIELD]
        elif row[LABEL_MAGNETIC_FIELD] > current_minimum_strength:
            data_frame.drop(index, inplace=True)


def polynomial_fit(data_frame: pandas.DataFrame, x_label: str, y_label: str,
                    degree: int) -> list[float]:
    """
    Fits a polynomial of a given degree to the data and returns the coefficients.
    """
    # Extract x and y values from the dataframe
    x = data_frame[x_label].values
    y = data_frame[y_label].values

    # Use numpy's polyfit function to fit a polynomial to the data
    coeffs = numpy.polyfit(x, y, degree)

    return coeffs.tolist()


def remove_outliers(data_frame: pandas.DataFrame, x_label: str, y_label: str,
                    coefficents:list[float], threshold: float = 3.0 ) -> pandas.DataFrame:
    """
    Removes points that are far away from the polynomial fit line.
    Returns the dataframe without outliers.
    Threshold is in standard deviations of the residuals.
    """
    # Extract x and y values from the dataframe
    x = data_frame[x_label].values
    y = data_frame[y_label].values
    # Calculate the predicted y values using the polynomial coefficients
    y_pred = numpy.polyval(coefficents, x)
    # Calculate the residuals (differences between the actual and predicted y values)
    residuals = y - y_pred
    # Calculate the standard deviation of the residuals
    std_residuals = numpy.std(residuals)
    # Determine the outlier threshold
    outlier_threshold = threshold * std_residuals
    # Identify non-outliers
    non_outliers = numpy.abs(residuals) <= outlier_threshold
    return data_frame[non_outliers]


def predict_value(slope:float, initial_x:int, initial_y:int, target_x:int) -> float:
    """
    Predicts the value of Y at a given X using the slope of the line.
    """
    return initial_y + slope*(target_x - initial_x)


def process_experiment_data(make_report:bool, polynomial_degree:int=1) -> float:
    """
    Preprocess the experiment data for each file in Raw folder.
    Removing background temperature, invalid magnetic field increases and outliers.
    Average slope of the line of best fit for each experiment is returned.
    """
    folder_path:str = os.path.dirname(os.path.realpath(__file__))  # Get the current files path
    folder_path = os.path.join(folder_path, r"Data\Raw")
    experiments_coefficients:list[float] = []
    average_coefficent:float = 0.0
    for i in range(1,len(os.listdir(folder_path))+1):
        data_frame = read_experiment_csv(i, True)
        if make_report:
            exploratory_analysis_report(data_frame)
        remove_background_temperature(data_frame)
        polynomial_coefficents = polynomial_fit(data_frame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, 1)
        graph_polynomial_fit(data_frame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_coefficents, title="Initial data with line of best fit for outlier removal")
        data_frame = remove_outliers(data_frame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_coefficents)
        remove_invalid_magnetic_increases(data_frame)
        polynomial_coefficents = polynomial_fit(data_frame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_degree)
        graph_polynomial_fit(data_frame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_coefficents, title=("Data after preprocessing with line of best fit: "+(', '.join(f"{num:.3f}" for num in polynomial_coefficents))))
        save_processed_csv(data_frame, i)
        experiments_coefficients.append(polynomial_coefficents[0])
    for coeff in experiments_coefficients:
        average_coefficent += coeff
    average_coefficent /= len(experiments_coefficients)
    return average_coefficent


def exploratory_analysis_report(data_frame: pandas.DataFrame) -> None:
    """
    Generate a pandas-profiling report for the data.
    Report will be saved as a HTML file in the Reports folder, with a number after the previous file.
    """
    file_path:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    report_number = 1
    file_path = os.path.join(file_path, "Reports")
    while os.path.isfile(os.path.join(file_path, "report"+str(report_number)+".html")):
        report_number += 1
    file_path = os.path.join(file_path, "report"+str(report_number)+".html")
    report = ProfileReport(data_frame)
    report.to_file(file_path)


if __name__ == "__main__":
    averageCoefficent = process_experiment_data(make_report=False)
    print("Average slope of the line of best fit across all experiments: "+ str(averageCoefficent))
    experimentDF = read_experiment_csv(5, False)
    predictX = experimentDF.iloc[0][LABEL_CURRENT]
    predictY = experimentDF.iloc[0][LABEL_MAGNETIC_FIELD]
    predictAt = experimentDF.iloc[-1][LABEL_CURRENT]
    predicted = predict_value(averageCoefficent, predictX, predictY, predictAt)
    ErrorPercentage = ((predicted - experimentDF.iloc[-1][LABEL_MAGNETIC_FIELD])/experimentDF.iloc[-1][LABEL_MAGNETIC_FIELD])*100
    print("Predicted value of magnetic field at: "+str(predictAt)+" is: "+str(predicted)+" with an error of: "+str(ErrorPercentage)+"%")
