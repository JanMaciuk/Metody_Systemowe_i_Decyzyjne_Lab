import numpy, pandas, matplotlib.pyplot, os


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
        dataFrame.to_csv(filePath)
    except Exception:
        print("save_processed_csv failed.") # We can continue execution, but the data will be lost.


def test() -> None:
    dataFrame = read_experiment_csv(1,True)
    print(dataFrame)
    save_processed_csv(dataFrame,1)

test()
