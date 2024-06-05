import pandas, matplotlib.pyplot, os

LABEL_MAGNETIC_FIELD:str = "Magnetic field strength B (µT)"
LABEL_CURRENT:str = "Current I (mA)"
LABEL_TIME:str = "Time t (s)"
LABEL_BACKGROUND_TEMP:str = "Background Temperature T (°C)"


def graph_data_against_time(dataFrame: pandas.DataFrame, current = True, magneticField = True, show=False) -> None:
    """
    Plot the experiment data from a DataFrame,
    Time is the x-axis, current and magnetic field on y-axis.
    """
    if not current and not magneticField:
        print("graph_data_against_time failed, no data selected to plot.")
        return
    if current:
        matplotlib.pyplot.plot(dataFrame[LABEL_TIME], dataFrame[LABEL_CURRENT],'o-', label=LABEL_CURRENT)
    if magneticField:
        matplotlib.pyplot.plot(dataFrame[LABEL_TIME], dataFrame[LABEL_MAGNETIC_FIELD], 'o-', label=LABEL_MAGNETIC_FIELD)
    matplotlib.pyplot.xlabel(LABEL_TIME)
    matplotlib.pyplot.ylabel("Current / Magnetic Field")
    matplotlib.pyplot.legend()
    if show:
        matplotlib.pyplot.show()
    else:
        #Save as Plot1, if it already exists save as Plot2, etc.
        filePath:str = os.path.dirname(os.path.realpath(__file__))  # Get the path of the current file
        plotNumber = 1
        filePath = os.path.join(filePath, "Plots")
        while os.path.isfile(os.path.join(filePath, "Plot"+str(plotNumber)+".png")):
            plotNumber += 1
        filePath = os.path.join(filePath, "Plot"+str(plotNumber)+".png")
        matplotlib.pyplot.savefig(filePath)
        matplotlib.pyplot.close()


