import pandas, matplotlib.pyplot, os, numpy

LABEL_MAGNETIC_FIELD:str = "Magnetic field strength B (µT)"
LABEL_CURRENT:str = "Current I (mA)"
LABEL_TIME:str = "Time t (s)"
LABEL_BACKGROUND_TEMP:str = "Background Temperature T (°C)"


def graph_data_against_time(dataFrame: pandas.DataFrame, current:bool = True, magneticField:bool = True, show:bool=False) -> None:
    """
    Plot the experiment data from a DataFrame,
    Time is the x-axis, current and magnetic field on y-axis.
    """
    if not current and not magneticField:
        print("graph_data_against_time failed, no data selected to plot.")
        return
    # Plot the data
    if current:
        matplotlib.pyplot.plot(dataFrame[LABEL_TIME], dataFrame[LABEL_CURRENT],'o-', label=LABEL_CURRENT)
    if magneticField:
        matplotlib.pyplot.plot(dataFrame[LABEL_TIME], dataFrame[LABEL_MAGNETIC_FIELD], 'o-', label=LABEL_MAGNETIC_FIELD)
    #Add labels and legend
    matplotlib.pyplot.xlabel(LABEL_TIME)
    matplotlib.pyplot.ylabel("Current / Magnetic Field")
    matplotlib.pyplot.legend()
    show_or_save(show)

def graph_polynomial_fit(dataFrame: pandas.DataFrame, x_label: str, y_label: str, 
                         coefficents: list[float], show:bool=False, title:str="Polynomial fit graph") -> None:
    """
    Plots the polynomial fit along with the original data points.
    """
    # Extract x and y values from the dataframe
    x = dataFrame[x_label].values
    y = dataFrame[y_label].values

    # Generate a range of x values for plotting the polynomial curve
    x_range = numpy.linspace(x.min(), x.max(), 500)
    # Calculate the corresponding y values using the polynomial coefficients
    y_fit = numpy.polyval(coefficents, x_range)

    # Plot the original data points
    matplotlib.pyplot.scatter(x, y, label='Data Points')
    # Plot the polynomial fit curve
    matplotlib.pyplot.plot(x_range, y_fit, label=f'Polynomial fit line')
    matplotlib.pyplot.gca().invert_xaxis()
    # Add labels and title
    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend()
    show_or_save(show)

def show_or_save(show:bool) -> None:
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