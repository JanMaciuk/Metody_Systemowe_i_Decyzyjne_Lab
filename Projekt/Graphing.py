import pandas, matplotlib.pyplot

LABEL_MAGNETIC_FIELD:str = "Magnetic field strength B (µT)"
LABEL_CURRENT:str = "Current I (mA)"
LABEL_TIME:str = "Time t (s)"
LABEL_BACKGROUND_TEMP:str = "Background Temperature T (°C)"


def graph_data_against_time(dataFrame: pandas.DataFrame, current = True, magneticField = True) -> None:
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
    matplotlib.pyplot.show()
