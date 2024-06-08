import numpy, pandas, os
from Graphing import LABEL_CURRENT, LABEL_MAGNETIC_FIELD, LABEL_TIME, LABEL_BACKGROUND_TEMP, graph_data_against_time, graph_polynomial_fit
from ydata_profiling import ProfileReport   # Newer version of pandas-profiling

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


def remove_background_temperature(dataFrame: pandas.DataFrame) -> None:
    """
    Remove the background temperature from the experiment data,
    returns a new DataFrame with the background temperature removed.
    """
    dataFrame = dataFrame.drop(LABEL_BACKGROUND_TEMP, axis='columns')


def remove_invalid_magnetic_increases(dataFrame: pandas.DataFrame) -> None:
    """
    Removes invalid increases in magnetic field strength from dataframe.
    Magnetic field should weaken with time as temeperature increases, an increase indicates invalid readings.
    Outliers should be removed before calling this function, they could introduce false minimas.
    """
    currentMinimumStrength = dataFrame.iloc[0][LABEL_MAGNETIC_FIELD]
    for index, row in dataFrame.iterrows():
        if row[LABEL_MAGNETIC_FIELD] < currentMinimumStrength:
            currentMinimumStrength = row[LABEL_MAGNETIC_FIELD]
        elif row[LABEL_MAGNETIC_FIELD] > currentMinimumStrength:
            dataFrame.drop(index, inplace=True)


def polynomial_fit(dataFrame: pandas.DataFrame, Xlabel: str, Ylabel: str, degree: int) -> list[float]:
    """
    Fits a polynomial of a given degree to the data and returns the coefficients.
    """
    # Extract x and y values from the dataframe
    x = dataFrame[Xlabel].values
    y = dataFrame[Ylabel].values

    # Use numpy's polyfit function to fit a polynomial to the data
    coeffs = numpy.polyfit(x, y, degree)

    return coeffs.tolist()


def remove_outliers(dataFrame: pandas.DataFrame, Xlabel: str, Ylabel: str, coefficents:list[float], threshold: float = 3.0 ) -> pandas.DataFrame:
    """
    Removes points that are far away from the polynomial fit line.
    Returns the dataframe without outliers.
    Threshold is in standard deviations of the residuals.
    """
    # Fit the polynomial and get the coefficients
    
    # Extract x and y values from the dataframe
    x = dataFrame[Xlabel].values
    y = dataFrame[Ylabel].values
    
    # Calculate the predicted y values using the polynomial coefficients
    y_pred = numpy.polyval(coefficents, x)
    
    # Calculate the residuals
    residuals = y - y_pred
    
    # Calculate the standard deviation of the residuals
    std_residuals = numpy.std(residuals)
    
    # Determine the outlier threshold
    outlier_threshold = threshold * std_residuals
    
    # Identify non-outliers
    non_outliers = numpy.abs(residuals) <= outlier_threshold
    return dataFrame[non_outliers]


def predict_value(slope:float, initialX:int, initialY:int, targetX:int) -> float:
    """
    Predicts the value of Y at a given X using the slope of the line.
    """
    return (initialY + slope*(targetX - initialX))


def Process_experiment_data(makeReport:bool, polynomialDegree:int=1) -> float:
    """
    Preprocess the experiment data for each file in Raw folder.
    Removing background temperature, invalid magnetic field increases and outliers.
    Average slope of the line of best fit for each experiment is returned.
    """
    folderPath:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    folderPath = os.path.join(folderPath, r"Data\Raw")
    ExperimentsCoefficients:list[float] = []
    averageCoefficent:float = 0.0
    for i in range(1,len(os.listdir(folderPath))+1):
        dataFrame = read_experiment_csv(i, True)
        if makeReport:
            exploratory_analysis_report(dataFrame)
        remove_background_temperature(dataFrame)
        polynomialCoefficents = polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, 1)
        graph_polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomialCoefficents, title="Initial data with line of best fit for outlier removal")
        dataFrame = remove_outliers(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomialCoefficents)
        remove_invalid_magnetic_increases(dataFrame)
        polynomialCoefficents = polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomialDegree)
        graph_polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomialCoefficents, title=("Data after preprocessing with line of best fit: "+(', '.join(f"{num:.3f}" for num in polynomialCoefficents))))
        save_processed_csv(dataFrame, i)
        ExperimentsCoefficients.append(polynomialCoefficents[0])
    for coeff in ExperimentsCoefficients:
        averageCoefficent += coeff
    averageCoefficent /= len(ExperimentsCoefficients)
    return averageCoefficent


def exploratory_analysis_report(dataFrame: pandas.DataFrame) -> None:
    """
    Generate a pandas-profiling report for the data.
    Report will be saved as a HTML file in the Reports folder, with a number after the previous file.
    """
    filePath:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
    reportNumber = 1
    filePath = os.path.join(filePath, "Reports")
    while os.path.isfile(os.path.join(filePath, "report"+str(reportNumber)+".html")):
        reportNumber += 1
    filePath = os.path.join(filePath, "report"+str(reportNumber)+".html")
    report = ProfileReport(dataFrame)
    report.to_file(filePath)


def test() -> None:
    dataFrame = read_experiment_csv(4,True)
    remove_background_temperature(dataFrame)
    #graph_data_against_time(dataFrame, current=False, magneticField=True)
    #graph_data_against_time(remove_outliers(dataFrame), current=False, magneticField=True)
    #print(polynomial_fit(dataFrame, LABEL_MAGNETIC_FIELD, LABEL_CURRENT, 2))

    #graph_data_against_time(dataFrame, current=True, magneticField=False, coefficents=polynomial_fit(dataFrame, LABEL_MAGNETIC_FIELD, LABEL_CURRENT, 2), show=True)
    polynomial_coefficents = polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, 1)
    graph_polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_coefficents)
    dataFrame = remove_outliers(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_coefficents)
    remove_invalid_magnetic_increases(dataFrame)
    graph_polynomial_fit(dataFrame, LABEL_CURRENT, LABEL_MAGNETIC_FIELD, polynomial_coefficents)

if __name__ == "__main__":
    averageCoefficent = Process_experiment_data(makeReport=False)
    print("Average slope of the line of best fit across all experiments: "+ str(averageCoefficent))
    experimentDF = read_experiment_csv(5, False)
    predictX = experimentDF.iloc[0][LABEL_CURRENT]
    predictY = experimentDF.iloc[0][LABEL_MAGNETIC_FIELD]
    predictAt = experimentDF.iloc[-1][LABEL_CURRENT]
    predicted = predict_value(averageCoefficent, predictX, predictY, predictAt)
    ErrorPercentage = ((predicted - experimentDF.iloc[-1][LABEL_MAGNETIC_FIELD])/experimentDF.iloc[-1][LABEL_MAGNETIC_FIELD])*100
    print("Predicted value of magnetic field at: "+str(predictAt)+" is: "+str(predicted)+" with an error of: "+str(ErrorPercentage)+"%")

#test()

#TODO:
# Use a linter to check for style issues.