import scipy.optimize as chisquarefit
import numpy as np
from Lightning import Lightning
import Medium
import copy

# Author: Ben Sheff
# Made for the Optics Group in Division XSD of the APS
# at Argonne National lab during the summer of 2015
#
# Working title Raven, a support class for the Optical Metrology ENgine (OMEN)
#
# Some of this documentation is a bit outdated at the moment
# This loops over all the available data files. For each one it reads in the
# data, parses it into a list of lists, and plots those lists.
# It then has a series of useful methods for fitting the data, both on the
# average curve and all the curves individually, and methods to remove the
# fit. There are also filters described in detail below.
#
# Important things to note / bugs
# The integrated data (heights), is done using simple left-sided Riemann
# summation. This leads to a resolution limitation, but it's to the same degree
# as the original slope data. There is also some error in the exact data
# positions as the x-positions delineate the leftmost end of the rectangle used
# to get the height listed at that point. In the case of the final point, it is
# a right-handed sum, as a left-handed one is impossible. It is of note that
# these errors can accumulate, as repeatedly recalculating slope and height
# will cause substantial distortions. The  differentiation code is roughly
# the inverse of the integraton code, but not exactly
# As of yet, the most methods in this and other classes assume all scans for
# the same segment have the same set of x-coordinates. Fixing this should be
# relatively simple, have the averaging methods also average the x-coordinates,
# and carry their x-coordinates with them. Then update all classes that use
# this to use x from the average data itself instead of raw data. You could
# do it more rigorously by interpolating the scans, and choosing one set of x
# coordinates at random to be the averaged version. The errors from this should
# be small. Empirically, they appear to be negligible
# There is some degree of error in the fitting functions. The polynomial fit
# is using a function explicitly designed to do polynomial fitting, and is
# quite good, but the circle one can be pretty aweful at times.
#
# Some methods:             * = may cause memory leaks due to use of copy class
# __init__(filename, num_scans) Reads in slope data from files derived from
#                               filename, iterating file indices in num_scans
# AcquireAnalysisData()(AAV)Calculates the Height data by integrating slopes
# AcquireSlopes()           Calculates Slope data by differentiating heights
# UpdateStats()             Gets avg/RMS of slope/height. Calls AnalyzeAverages
# AnalyzeAverages()         Gets avg/RMS/peak-to-valley of slope/height/residue
# ResetFilter()             Sets the list of scans passing filter to all scans
# ApproveFilter()           Removes the scans not in filtered_list (not passing
#                               the filter) from the data
# Reset()                   Recalculates averages, RMS's, and filters
# SetUndo()*                Sets backups for slopes, heights, and scan_list*
# Undo()                    Uses backups from SetUndo
# SetStartPoint()*          Sets second backup for slopes, heights, scan_list*
# ReturnToStartPoint()      Uses backups from SetStartPoint
# SetROI(list region=[start, end]) Removes any data with x-value outside region
# FitCircle(data_type)      Fits a circle to data clarified in data_type
# FitPolynomial(degree, data_type) Fits a degree degree polynomial to data_type
# UpdateResidue()*          Recalculates residue, subtracting the fit from data
#                               chosen in fit_data_type, and gets related stats
# SubtractFit()             Sets data from fit_data_type to the current residue
# PrepFilter()              Finds integrated positive deviation in standard
#                               deviations beyond 1 for slope and height data
# FilterHeights(rigor)      Removes any scan with height_deviation above the
#                               limit from filtered_list. Updates filter_rigor
#                               deviations beyond 1 for slope and height data
# FilterSlopes(rigor)       Removes any scan with slope_deviation above the
#                               limit from filtered_list. Updates filter_rigor
# AddCalibration(raven)     Updates all internal data to reflect a given
#                               calibration curve, including fit.
# Save(filename)            Saves slopes, heights, fit_function, fit_avg, and
#                               scan_list to filename so work can be recreated
# SaveAvg(filename)         Saves only the position and data for the active avg
# Load(filename)            Loads data from either raw files or OMEN-made ones
# GetSlopes()               Returns those slope traces with indices in
#                               filtered_list, after adding the offset
# GetHeights()              Returns those height traces with indices in
#                               filtered_list, after adding the offset
# GetScanList()             Returns scan indices with indices in filtered_list
# GetResidue()              Returns residue traces with indices in
#                               filtered_list, after adding the offset
# SetOffset()               Sets the offset to be applied to all data
#
#
# Note variables have further clarification in the class itself.
# Useful Variables:         Created in:         Purpose:
# default_file              __init__            Initial file input for data
# slope_sigmas              __FindSlopeRMSOverScans         RMS of position
#                                                       over scans
# slope_avg                 __FindSlopeAverageOverScans Avg position over scans
# height_sigmas             __FindHeightRMSOverScans        RMS integrated data
#                                                   at each position over scans
# height_avg                __FindHeightAverageOverScans    Avg integrated data
#                                                   at each position over scans
# scan_list                 __init__            The original indices of scans
# slope_results             AnalyzeAverages     Data about the avg slope trace
# height_results            AnalyzeAverages     Data about the avg height trace
# FOO_last where x is any   SetUndo             Holds old data before recent
#   variable above                                  action, so you can undo
# FOO_initial where x is    SetStartPoint       Allows for easier large-scale
#   any variable above                              undo-ing of actions
# fit_function              FitFOO              Data for plot of fits
# fit_avg                   FitFOO              As fit_function, but fit to avg
# read_fit                  FitFOO              Printable for fitted function
# fit_data_type             FitFOO              Records which data is fitted
# residue                   UpdateResidue       Holds active data minus fit
# residue_avg               __FindResidueAverageOverScans   Avg of the residue
#                                                   over scans
# residue_sigmas            __FindResidueRMSOverScans       RMS of the residue
#                                                   over scans
# residue_results           AnalyzeAverages/UpdateResidue   Data about the avg
#                                                   residue trace
# slope_deviation           PrepFilter          Holds level of deviation: slope
#                                                   each run, for faster filter
# height_deviation          PrepFilter          Holds level of deviation height
#                                                   each run, for faster filter
# filter_rigor              FilterFOO           The level of filtering, allowed
#                                                   deviation is inverse
# filtered_list             FilterFOO/ResetFilter   Holds indices in current
#                                                   data list of allowed runs
# radius                    FitCircle           Radius of mirror, from avg run
# offset                    SetOffset           The offset applied to all data
#
#
# Finally, private variables that are still pretty important:
# Variable:                 Created in:         Purpose:
# __slopes                  __init__            All the raw data
# __heights                 AAV                 Integrated raw data


class Raven:
    # Setting up some attributes.
    __slopes = []   # Format is [ScanNumber, {0:x (followed by index), 1:y
    #                  (followed by index)]
    __slopes_last = []  # Format is [ScanNumber, {0:x (followed by index), 1:y
    #                  (followed by index)]
    __slopes_initial = []  # Format is [ScanNumber, {0:x (followed by index),
    #                   1:y (followed by index)]
    __heights = []  # Format is [ScanNumber, {0:x, 1:y}]
    __heights_last = []  # Format is [ScanNumber, {0:x, 1:y}]
    __heights_initial = []  # Format is [ScanNumber, {0:x, 1:y}]
    slope_sigmas = []  # Format is [position-index]
    slope_avg = []  # Format is [position-index]
    height_sigmas = []  # Format is [position-index]
    height_avg = []  # Format is [position-index]
    fit_data_type = 0  # Format is {1:height data, 0:slope data}
    residue = []   # Format is [ScanNumber, {0:x, 1:y] since I use the term a
    #                   lot, I'll clarify again, residue = data minus fit
    residue_avg = []  # Format is [position-index]
    residue_sigmas = []  # Format is [position-index]
    slope_results = []  # Format is [{0:average, 1:RMS, 2:p-v}]
    height_results = []  # Format is [{0:average, 1:RMS, 2:p-v}]
    residue_results = []  # Format is [{0:average, 1:RMS, 2:p-v}]
    default_file = "C:\\Users\\Ben Sheff\\Documents\\Argonne\\"
    # default_file += "Test Data\\Sheff_300mm_test0001_01_1.asc"
    #default_file += "\\Flat_Mirror_Segments\\Flat_Mirror0001_01_1.asc"
    default_file += "\\Flat_Mirror_Full\\Flat_Mirror_full_03_1.asc"
    scan_list = list(range(99))  # Format is [index]
    scan_list_last = list(range(99))  # Format is [index]
    scan_list_initial = list(range(99))  # Format is [index]
    fit_function = []  # Format is [ScanNumber, {0:x (followed by index), 1:y
    #                  (followed by index)]
    fit_function_last = []  # Format is [ScanNumber, {0:x (followed by index),
    #                  1:y (followed by index)]
    fit_function_initial = []  # Format is [ScanNumber, {0:x (followed by
    #                  index), 1:y (followed by index)]
    fit_avg = []  # Format is [{0:x (followed by index), 1:y
    #                  (followed by index)]
    read_fit = ""  # Format is whatever seems good for reading the function
    slope_deviation = []  # Format is [ScanNumber]
    height_deviation = []  # Format is [ScanNumber]
    filter_rigor = 0.0000000001  # Format is positive real number
    filtered_list = list(range(99))  # Format is [index]
    radius = 0  # Format is just a number
    offset = [0, 0]  # Format is [x-offset, y-offset]

    __current_value = -1  # This is just to account for the way python does scope
    __use_fit_avg = False  # This toggles if the subtracted fit is average or
    #                        run-by-run fitting

    # Initializing the object by grabbing the files and reading out slopes. It
    # requires filename to be a format string to which it adds a 2 digit number
    def __init__(self, filename=default_file, num_scans=scan_list):
        if filename == []:
            return
        self.Load(filename, num_scans)

    # The name is a relic of older, more compact versions of this software.
    # This method really just integrates the slope to get height data, and sets
    # up basic stats by the Reset() method
    def AcquireAnalysisVariables(self):
        self.SetUndo()
        self.__heights = []
        __slope_avg_by_scan = [0 for scan in self.__slopes]

        # First we need to get the integrated data. This works best if we first
        # remove the average of the slope data, so:
        for scan in enumerate(self.__slopes):
            for y_slope in enumerate(scan[1][1]):
                x_range = len(self.__slopes[scan[0]][0])
                __slope_avg_by_scan[scan[0]] += y_slope[1] / x_range

        # This loop will find the height data by integrating slope data after
        # subtracting the average slope
        for scan in enumerate(self.__slopes):
            average = __slope_avg_by_scan[scan[0]]
            integration = 0.0
            riemann_width = 0.0
            integrated_scan = []
            for y_slope in enumerate(np.delete(scan[1][1], len(scan[1][1])-1)):
                riemann_width = (scan[1][0][y_slope[0]+1] -
                                 scan[1][0][y_slope[0]])
                integration += ((y_slope[1]-average)*riemann_width)
                integrated_scan.append(integration)
            # To match up the x-coordinates, we now add a right-handed riemann
            # square to fill out the last coordinate
            temp = len(scan[1][1])-1
            integration += (scan[1][1][temp]-average)*riemann_width
            integrated_scan.append(integration)
            self.__heights.append([list(scan[1][0]), integrated_scan])

        # Now for the some averages and RMS's for height and slope. In
        # previous versions of this code, these integrated together for speed
        # (since they all do the same loops over and over), but this causes
        # problems, and doesn't save all that much time.
        self.Reset()

    # Counterpart to AcquireAnalysisVariables. It differentiates heights by
    # taking the height at two adjacent points, and dividing the difference
    # by the difference in position. It is left hand focused, so each point is
    # the slope to the one on the right, except the final point, which is
    # identical to the penultimate one.
    def AcquireSlopes(self):
        self.SetUndo()
        self.__slopes = []

        # This loop will find the slope data by differentiating slope data
        for scan in enumerate(self.__heights):
            riemann_width = 0.0
            differentiated_scan = []
            print(scan[1][1][0], scan[1][0][1]-scan[1][0][0])
            derivative = scan[1][1][0] / (scan[1][0][1] - scan[1][0][0])
            differentiated_scan.append(derivative)
            for index in range(len(scan[1][1]) - 2):
                riemann_width = (scan[1][0][index+2] -
                                 scan[1][0][index+1])
                derivative = ((scan[1][1][index+1] - scan[1][1][index])
                              / riemann_width)
                differentiated_scan.append(derivative)

            # To match up the x-coordinates, we now add a right-handed slope
            # to fill out the last coordinate
            temp = len(scan[1][1])-1
            derivative = (scan[1][1][temp]-scan[1][1][temp-1])/riemann_width
            differentiated_scan.append(derivative)
            self.__slopes.append([list(scan[1][0]), differentiated_scan])

        # Now for the some averages and RMS's for height and slope. In
        # previous versions of this code, these integrated together for speed
        # (since they all do the same loops over and over), but this causes
        # problems, and doesn't save all that much time.
        self.Reset()

    # Recalculating all of the averages, RMS's, and the stats for average run.
    def UpdateStats(self):
        self.__FindHeightAverageOverScans()
        self.__FindHeightRMSOverScans()
        self.__FindSlopeAverageOverScans()
        self.__FindSlopeRMSOverScans()
        self.AnalyzeAverages()

    # Resets the list of runs passing the filter to be all runs
    def ResetFilter(self):
        self.filtered_list = list(range(len(self.__slopes)))

    # Sets the data to just those passed the filter, by only keeping runs with
    # indices in filtered_list
    def ApproveFilter(self):
        self.SetUndo()
        self.scan_list = [self.scan_list[i] for i in self.filtered_list]
        self.__slopes = [self.__slopes[i] for i in self.filtered_list]
        self.__heights = [self.__heights[i] for i in self.filtered_list]
        self.Reset()

    # A useful method to recalculate everything after you make a change.
    # Specifically, this does the residue, RMS/avg of runs, stats of avg run,
    # and the deviations of each run's slope and height.
    def Reset(self):
        self.UpdateResidue()
        self.UpdateStats()
        self.ResetFilter()
        self.PrepFilter()

    # This method saves an old version of the data from before the last action
    def SetUndo(self):
        self.__slopes_last = copy.deepcopy(self.__slopes)
        self.__heights_last = copy.deepcopy(self.__heights)
        self.fit_function_last = copy.deepcopy(self.fit_function)
        self.scan_list_last = copy.deepcopy(self.scan_list)

    # Enacting the undo
    def Undo(self):
        self.__slopes = self.__slopes_last
        self.__heights = self.__heights_last
        self.fit_function = copy.deepcopy(self.fit_function_last)
        self.scan_list = self.scan_list_last
        self.Reset()
        self.SetUndo()

    # Setting a long-term reference point
    def SetStartPoint(self):
        self.__slopes_initial = copy.deepcopy(self.__slopes)
        self.__heights_initial = copy.deepcopy(self.__heights)
        self.fit_function_initial = copy.deepcopy(self.fit_function)
        self.scan_list_initial = copy.deepcopy(self.scan_list)

    # Using the long-term reference point
    def ReturnToStartPoint(self):
        self.__slopes = self.__slopes_initial
        self.__heights = self.__heights_initial
        self.fit_function = copy.deepcopy(self.fit_function_initial)
        self.scan_list = self.scan_list_initial
        self.Reset()
        self.SetStartPoint()

    # Method to set an offset to apply to all outputs from this class
    def SetOffset(self, xoffset=0, yoffset=0):
        self.offset = [xoffset, yoffset]

    # Method to define a region of interest, and cut off outside data
    def SetROI(self, region):
        self.SetUndo()
        if self.__slopes == []:
            return
        if self.__slopes[0] == []:
            return
        indices_to_remove = []
        for index in range(len(self.__slopes[0][0])):
            if self.__slopes[0][0][index] < region[0]:
                indices_to_remove.append(index)
            if self.__slopes[0][0][index] > region[1]:
                indices_to_remove.append(index)
        for scan in self.__slopes:
            for i in reversed(indices_to_remove):
                del scan[0][i]
                del scan[1][i]
        temp = indices_to_remove.pop()
        if temp != len(self.__heights[0][0]):
            indices_to_remove.append(temp)
        for scan in self.__heights:
            for i in reversed(indices_to_remove):
                del scan[0][i]
                del scan[1][i]
        self.Reset()

    # Method to set some x-value as zero and shift x accordingly
    def ZeroX(self, zero=0):
        self.SetUndo()
        if self.__slopes == []:
            return
        if self.__slopes[0] == []:
            return
        if zero == 0:
            temp = self.__slopes[0][0]
            zero = (temp[len(temp)-1]-temp[0])/2
        thingstozero = [self.__slopes, self.__heights,
                        self.fit_function, self.fit_avg]
        for data in thingstozero:
            for ScanNumber in range(len(data)):
                try:
                    for position in range(len(list(data[ScanNumber][0]))):
                        data[ScanNumber][0][position] -= zero
                except TypeError:
                    if ScanNumber == 0:
                        for position in range(len(list(data[0]))):
                            data[0][position] -= zero
        self.Reset()

    # Fits circles to data and the average of the data
    # data_type=1 means fit to height, data_type=0 means fit to slope
    # Currently it fits by two iterations, but the fitter is not very good.
    # It tends to fall fairly easily into local minima, or rail to large radius
    # Could be fixed by hardcoding initial guesses based on three widely spaced
    # points in data_avg
    def FitCircle(self, data_type=__current_value):
        if data_type == -1:
            data_type = self.fit_data_type
        else:
            self.fit_data_type = data_type
        data = []
        data_avg = []
        if data_type == 1:
            data = self.__heights
            data_avg = self.height_avg
            data_results = self.height_results
        if data_type == 0:
            data = self.__slopes
            data_avg = self.slope_avg
            data_results = self.slope_results
        self.fit_function = []
        self.read_fit = ""

        for scan in data:
            fit = []
            x0 = scan[0][int(len(scan[0])/2)]
            y0 = 50/(data_results[2])
            if y0 > 100000:
                y0 = 100000
                
            R = y0
            while True:
              try:
                fit1 = chisquarefit.curve_fit(self.__circle, scan[0],
                                              scan[1], [x0, y0, R])
                fit2 = chisquarefit.curve_fit(self.__negcircle, scan[0],
                                              scan[1], [x0, y0, R])

                chiR1 = np.sqrt(abs(np.diag(fit1[1])))[1]
                chiR2 = np.sqrt(abs(np.diag(fit2[1])))[1]
                if chiR1 <= chiR2:
                    fit = [chisquarefit.curve_fit(self.__circle, data[0][0],
                                                  data_avg, fit1[0]), fit2, 0]
                if chiR2 < chiR1:
                    fit = [chisquarefit.curve_fit(self.__negcircle, data[0][0],
                                                  data_avg, fit2[0]), fit1, 1]

                if min(chiR1, chiR2) > 1000000000:
                    raise RuntimeError("Fit had a problem")

                fit_y = [pow(-1, fit[2])*self.__circle(x, fit[0][0][0],
                                                       fit[0][0][1], fit[0][0][2])
                         for x in scan[0]]
                self.fit_function.append([copy.deepcopy(scan[0]), fit_y])
                break
              except RuntimeError:
                y0 *= 10

        x0 = scan[0][int(len(data[0][0])/2)]
        y0 = abs(y0)
        R = y0
        fit = []
        fit1 = chisquarefit.curve_fit(self.__circle, data[0][0],
                                      data_avg, [x0, y0, R])
        fit2 = chisquarefit.curve_fit(self.__negcircle, data[0][0],
                                      data_avg, [x0, y0, R])
        chiR1 = np.sqrt(abs(np.diag(fit1[1])))[1]
        chiR2 = np.sqrt(abs(np.diag(fit2[1])))[1]
        if chiR1 <= chiR2:
            fit = [chisquarefit.curve_fit(self.__circle, data[0][0],
                                          data_avg, fit1[0]), fit2, 0]
        if chiR2 < chiR1:
            fit = [chisquarefit.curve_fit(self.__negcircle, data[0][0],
                                          data_avg, fit2[0]), fit1, 1]
        sign = pow(-1, fit[2])
        fit_y = [sign*self.__circle(x, fit[0][0][0], fit[0][0][1],
                                    fit[0][0][2]) for x in data[0][0]]
        self.fit_avg = [copy.deepcopy(data[0][0]), fit_y]
        self.read_fit = "sqrt({0:.6g}^2 - (x-{1:.6g})^2) - {2:.6g}"
        self.read_fit = self.read_fit.format(fit[0][0][2], fit[0][0][0],
                                             fit[0][0][1])
        self.radius = fit[0][0][2]
        self.UpdateResidue()
        return self.read_fit

    # Fits given degree polynomials to data and average of the data,
    # data_type=1 means fit to height, data_type=0 means fit to slope
    def FitPolynomial(self, degree=0, data_type=__current_value):
        if data_type == -1:
            data_type = self.fit_data_type
        else:
            self.fit_data_type = data_type
        data = []
        data_avg = []
        if data_type == 1:
            data = self.__heights
            data_avg = self.height_avg
        if data_type == 0:
            data = self.__slopes
            data_avg = self.slope_avg
        self.read_fit = ""

        self.fit_function = []
        for i, scan in enumerate(data):
            self.fit_function.append([copy.deepcopy(scan[0]), []])
            fit = np.poly1d(np.polyfit(scan[0], scan[1], degree))
            for datum in scan[0]:
                self.fit_function[i][1].append(fit(datum))

        self.fit_avg = [copy.deepcopy(data[0][0]), []]
        coefs = np.polyfit(data[0][0], data_avg, degree)
        fit = np.poly1d(coefs)
        self.fit_avg[1] = [fit(x) for x in data[0][0]]
        coefs = list(coefs)
        constant = coefs.pop(degree)
        for i, coef in enumerate(coefs):
            self.read_fit += "{0:.6g}x^{1:d} + ".format(coef, degree-i)
        self.read_fit += "{0:.6g}".format(constant)
        if degree == 2:
            self.radius = 1 / (2 * coefs[0])
        self.UpdateResidue()
        return self.read_fit

    def UseFitAverage(self, use_fit_avg="toggle"):
        if use_fit_avg == "toggle":
            self.__use_fit_avg = not self.__use_fit_avg
        else:
            self.__use_fit_avg = use_fit_avg

    # Method to remove subtract the fit and save the residue in residue
    def UpdateResidue(self):
        data = []
        if self.fit_data_type == 1:
            data = self.__heights
        if self.fit_data_type == 0:
            data = self.__slopes
        self.residue = copy.deepcopy(data)
        for scan in range(len(data)):
            for pos in range(len(data[scan][1])):
                if self.__use_fit_avg:
                    fit = self.fit_avg[1]
                else:
                    fit = self.fit_function[scan][1]
                self.residue[scan][1][pos] -= fit[pos]
        self.__FindResidueAverageOverScans()
        self.__FindResidueRMSOverScans()
        self.residue_results = self.__FindResults(self.residue_avg)

    # Once you like your fit, this subtracts it from the data
    def SubtractFit(self):
        self.UpdateResidue()
        self.SetUndo()
        if self.fit_data_type == 1:
            self.__heights = self.residue
            self.FilterHeights()
        if self.fit_data_type == 0:
            self.__slopes = self.residue
            self.FilterSlopes()
        self.Reset()

    # Finds the RMS of the slope over all scans at a given position
    def __FindSlopeRMSOverScans(self):
        self.slope_sigmas = np.array(range(len(self.__slopes[0][0])))*0.
        for scan in self.__slopes:
            for y_slope in enumerate(scan[1]):
                temp = (y_slope[1]-self.slope_avg[y_slope[0]])**2
                self.slope_sigmas[y_slope[0]] += temp/len(self.__slopes)
        self.slope_sigmas = list(np.sqrt(self.slope_sigmas))

    # Finds the RMS of the height over all scans at a given position
    def __FindHeightRMSOverScans(self):
        temp = len(self.__heights[0][1])
        self.height_sigmas = np.array(range(temp))*0.
        for scan in self.__heights:
            for y_slope in enumerate(scan[1]):
                temp = (y_slope[1]-self.height_avg[y_slope[0]])**2
                self.height_sigmas[y_slope[0]] += temp/len(self.__heights)
        self.height_sigmas = list(np.sqrt(self.height_sigmas))

    # Finds the RMS of the residue over all scans at a given position
    def __FindResidueRMSOverScans(self):
        temp = len(self.residue[0][1])
        self.residue_sigmas = np.array(range(temp))*0.
        for scan in self.residue:
            for y_slope in enumerate(scan[1]):
                temp = (y_slope[1]-self.residue_avg[y_slope[0]])**2
                self.residue_sigmas[y_slope[0]] += temp/len(self.residue)
        self.residue_sigmas = list(np.sqrt(self.residue_sigmas))

    # Finds the average residue over all scans at a given position
    def __FindResidueAverageOverScans(self):
        self.residue_avg = [0 for residue in self.residue[0][0]]
        for scan in self.residue:
            for residue in enumerate(scan[1]):
                self.residue_avg[residue[0]] += residue[1] / len(self.residue)

    # Finds the average height over all scans at a given position
    def __FindHeightAverageOverScans(self):
        self.height_avg = [0 for height in self.__heights[0][0]]
        for scan in self.__heights:
            for height in enumerate(scan[1]):
                self.height_avg[height[0]] += height[1] / len(self.__heights)

    # Finds the average slope over all scans at a given position
    def __FindSlopeAverageOverScans(self):
        self.slope_avg = [0 for slope in self.__slopes[0][0]]
        for scan in self.__slopes:
            for slope in enumerate(scan[1]):
                self.slope_avg[slope[0]] += slope[1] / len(self.__slopes)

    # Finds the average, RMS, and peak-to-valley of the chosen averaged data
    def __FindResults(self, averaged_data):
        length = len(averaged_data)
        average = 0.0
        std = 0.0
        p2v = averaged_data[0]
        valley = p2v
        for datum in averaged_data:
            average += datum / length
            if datum > p2v:
                p2v = datum
        for datum in averaged_data:
            std += (datum - average)**2 / length
            if valley > datum:
                valley = datum
        p2v -= valley
        std = np.sqrt(std)
        return [average, std, p2v]

    # Removing outlier datasets:

    # This method takes the height at each point, and if that is more than a
    # standard deviation out, it takes how many standard deviations out it is
    # and adds that to a tally of total deviation for the curve. If this is
    # beyond the number (number of points in scan)/(7*inputted rigor), then
    # the scan is cut from the list of allowed scans' indices (filtered_list)
    # in FilterFOO
    def PrepFilter(self):
        self.height_deviation = []
        self.slope_deviation = []
        for scan in self.__heights:
            counter = 0
            for x in range(len(scan[1])):
                if (abs(scan[1][x]-self.height_avg[x])
                        > (1.*self.height_sigmas[x])):
                    counter += ((abs(scan[1][x]-self.height_avg[x]) -
                                 (1.*self.height_sigmas[x]))
                                / self.height_sigmas[x])
            self.height_deviation.append(counter)
        for scan in self.__slopes:
            counter = 0
            for x in range(len(scan[1])):
                if (abs(scan[1][x]-self.slope_avg[x])
                        > (1.*self.slope_sigmas[x])):
                    counter += (abs(scan[1][x]-self.slope_avg[x]) -
                                (1.*self.slope_sigmas[x]))/self.slope_sigmas[x]
            self.slope_deviation.append(counter)

    # Takes the deviations found in height_deviation and compares them to a
    # limit, and cuts out indices of any found too deviant from list of allowed
    # more clarification in PrepFilter
    def FilterHeights(self, rigor=__current_value):
        if rigor == -1:
            rigor = self.filter_rigor
        else:
            self.filter_rigor = rigor
        self.ResetFilter()
        max_deviation = len(self.__slopes[0][0]) / (7*rigor)
        for i, deviation in reversed(list(enumerate(self.height_deviation))):
            if deviation > max_deviation:
                del self.filtered_list[i]

    # Takes the deviations found in slope_deviation and compares them to a
    # limit, and cuts out indices of any found too deviant from list of allowed
    # more clarification in PrepFilter
    def FilterSlopes(self, rigor=__current_value):
        if rigor == -1:
            rigor = self.filter_rigor
        else:
            self.filter_rigor = rigor
        self.ResetFilter()
        max_deviation = len(self.__slopes[0][0]) / (7*rigor)
        for i, deviation in reversed(list(enumerate(self.slope_deviation))):
            if deviation > max_deviation:
                del self.filtered_list[i]

    # Circle Functions for fitting above
    def __circle(self, x, x0, y0, R):
        out = (R * 1000)**2 - ((x-x0) * 0.001)**2
        # out += (abs(out) - out)*123412345678909876543212345 # This is so tiny
        #                                                 circles don't happen.
        out = (abs(out) + out) * 0.5
        return -1000000000*(np.sqrt(out) - y0)

    # Negative of the above circle function, since I was getting errors when I
    # tried to fit negative __circle()
    def __negcircle(self, x, x0, y0, R):
        return -1 * self.__circle(x, x0, y0, R)

    # Method to find the RMS, peak to peak, and average of the average scan
    def AnalyzeAverages(self):
        self.slope_results = self.__FindResults(self.slope_avg)
        self.height_results = self.__FindResults(self.height_avg)
        self.residue_results = self.__FindResults(self.residue_avg)

    # This will add another Raven file as a calibration curve, and apply it. It
    # assumes the calibration curve is for slope data, and recalculates height
    # data appropriately. It also blindly applies the same to the fit function,
    # refitting is recommended in whatever is using this file.
    def AddCalibration(self, raven):
        self.SetUndo()
        calibration_curve = [raven.GetSlopeAverage(), raven.GetSlopes()[0][0]]
        calibration_curve[0].insert(0, raven.GetSlopes()[0][0][0]-1)
        calibration_curve[1].insert(0, raven.GetSlopes()[0][0][0]-1)
        calibration_curve[0].insert(0, -1000)
        calibration_curve[1].insert(0, -1000)
        length = len(raven.GetSlopes()[0][0])-1
        calibration_curve[0].append(raven.GetSlopes()[0][0][length])
        calibration_curve[1].append(raven.GetSlopes()[0][0][length])
        calibration_curve[0].append(1000)
        calibration_curve[1].append(1000)
        interp_curve = Lightning(calibration_curve)
        self.__slopes = [[scan[0],
                          interp_curve.LinearInterpolate(scan[1], 0, 0)]
                         for scan in self.__slopes]
        self.fit_function = [[scan[0],
                              interp_curve.LinearInterpolate(scan[1], 0, 0)]
                             for scan in self.__slopes]
        self.fit_avg = [self.fit_avg[0],
                        interp_curve.LinearInterpolate(self.fit_avg[1], 0, 0)]
        self.AcquireAnalysisVariables()

    # Method to save everything
    def Save(self, filename=__current_value):
        if filename == -1:
            filename = self.default_file.format(0).split(".as")[0]+"_out.asc"
        fileInQuestion = open(filename, "w")
        output = []
        output.append(copy.deepcopy(self.__slopes))
        for i, scan in enumerate(output[0]):
            if i == 0:
                scan[0].insert(0, "# X-position")
                scan[1].insert(0, "Slope, scan " + str(i))
            else:
                del scan[0]
                scan[0].insert(0, "Slope, scan " + str(i))

        output.append(copy.deepcopy(self.__heights))
        for i, scan in enumerate(output[1]):
            del scan[0]
            scan[0].insert(0, "Height, scan " + str(i))

        output.append(copy.deepcopy(self.fit_function))
        for i, scan in enumerate(output[2]):
            del scan[0]
            scan[0].insert(0, "Fitted Function Value, scan " + str(i))

        output.append(copy.deepcopy(self.fit_avg))
        del output[3][0]
        output[3][0].insert(0, "Fitted Function to Average")

        output_string = "# This is a bulk file for use by OMEN\n"
        output_string += "# List of Available Scans: "
        for scan_number in self.scan_list:
            output_string += str(scan_number) + ", "
        output_string += "\n# Internal Note: " + str(self.offset[0]) + " "
        output_string += str(self.offset[1]) + " " + str(self.fit_data_type)
        output_string += "\n\n\n"

        output_string += Medium.ConvertToAscii(output)

        fileInQuestion.write(output_string)

    # Method to save just the average, active data
    def SaveAvg(self, filename=__current_value):
        if filename == -1:
            filename = self.default_file.format(0).split(".as")[0]+"_avg.asc"
        fileInQuestion = open(filename, "w")
        output = []
        if self.fit_data_type == 1:
            output.append(copy.deepcopy(self.GetHeights()[0][0]))
            output.append(copy.deepcopy(self.GetHeightAverage()))
            output[1].insert(0, "# Mirror Height")
        if self.fit_data_type == 0:
            output.append(copy.deepcopy(self.GetSlopes()[0][0]))
            output.append(copy.deepcopy(self.GetSlopeAverage()))
            output[1].insert(0, "# Mirror Slope")
        output[0].insert(0, "# X-position")
        output_string = "# This is an averaged file generated by OMEN\n"
        output_string += "# Internal Note: " + str(self.offset[0]) + " "
        output_string += str(self.offset[1]) + " " + str(self.fit_data_type)
        output_string += "\n\n\n"
        output_string += Medium.ConvertToAscii(output)

        fileInQuestion.write(output_string)

    # Method to load data from a file
    def Load(self, filename=__current_value, num_scans=scan_list):
        if filename == -1:
            filename = self.default_file.format(0).split(".as")[0]+"_out.asc"
        try:
            fileInQuestion = open(filename.format(1), "r")
            # Based on the first line, it detrmines the file format
            currentline = fileInQuestion.readline()
            temp = "# This is a"  # Sadly a needed line for style guidelines
            if not(currentline.startswith("## mda2ascii ") or
                   currentline == temp+" bulk file for use by OMEN\n" or
                   currentline == temp+"n averaged file generated by OMEN\n"):                raise FileNotFoundError("Bad File")
        except FileNotFoundError:
            return 0
        except AttributeError:
            currentline = "not a file"

        output = 0
        # Clearing house and preparing the defaults
        self.clear()
        try:
            self.default_file = filename.split("_out.as")[1]
            self.default_file = filename.split("_out.as")[0]
        except:
            self.default_file = filename

        # This assumes any file starting with "## mda2ascii " came from the LTP
        if currentline.startswith("## mda2ascii "):
            name = list(filename)
            del name[len(name) - 7]
            del name[len(name) - 7]
            name.insert(len(name) - 6, "{0:02d}")
            formatted_name = ""
            for char in name:
                formatted_name += char
            self.__LoadRawFile(formatted_name, num_scans)
            output = 1

        # Since I have more control over my format, this is more specific,
        # anything starting with my opening lines is an OMEN format file.
        temp = "# This is a"
        if (currentline == temp + " bulk file for use by OMEN\n" or
                currentline == temp + "n averaged file generated by OMEN\n"):
            self.__LoadBulkFile(fileInQuestion)
            output = 2
        if currentline == temp + "n averaged file generated by OMEN\n":
            output = 3

        # This is if a user inputs a list instead of a file
        if currentline == "not a file":
            self.__slopes.append(copy.deepcopy(filename))
            self.__heights.append(copy.deepcopy(filename))
            self.fit_function.append(copy.deepcopy(filename))
            self.default_file = ""

        self.Reset()
        self.SetUndo()
        self.SetStartPoint()

        return output

    # Private Method to read in data outputted by the EPICS LTP software
    def __LoadRawFile(self, filename=__current_value, num_scans=scan_list):
        if filename == -1:
            filename = self.default_file
        self.scan_list = list(num_scans)
        bad_indices = []
        for index, j in enumerate(self.scan_list):
            try:
                fileInQuestion = open(filename.format(j+1), "r")
                output = "\n"

                ltp_data_location = 4
                ltp_data_text = ""
                # This loops is to get past header text. If any operations
                # using the header text are needed, insert them here.
                while((output.startswith("#") or output.startswith("\n")) or
                       output.startswith("     ")):
                    output = fileInQuestion.readline()
                    if output.endswith("ltp:ElcomatAM1:Yavg.VAL, Y Average, microra\n"):
                        ltp_data_text = output

                if ltp_data_text != "":
                    ltp_data_text = ltp_data_text.split(" ")
                    index = 0
                    for possible_value in ltp_data_text:
                        try:
                            temp = int(possible_value) - 1
                            ltp_data_location = temp
                            break
                        except ValueError:
                            continue

                # This loop is acquiring the actual data
                prelim_data = []
                while(output.endswith("\n")):
                    prelim_data.append(output.split("\n")[0].split(" "))
                    output = fileInQuestion.readline()
                intermediate_step = np.array(prelim_data, dtype=list)
                data = intermediate_step.astype(np.float).T
                x = list(data[1])
                y = list(data[ltp_data_location])

                # Time to save this to the master list
                self.__slopes.append([x, y])
                fileInQuestion.close()
            except FileNotFoundError:
                # Remove the bad files
                bad_indices.append(index)
        for index in reversed(bad_indices):
            del self.scan_list[index]

        self.fit_data_type = 0
        if len(self.__slopes) > 0:
            self.fit_function = self.__slopes
            self.AcquireAnalysisVariables()

    # Private Method to read in data created by OMEN
    def __LoadBulkFile(self, fileInQuestion):
        self.scan_list = [0]
        output = ""
        currentline = fileInQuestion.readline()
        while currentline.endswith("\n"):
            if currentline.startswith("#") or currentline.startswith("\n"):
                if currentline.startswith("# List of Available Scans: "):
                    temp = currentline.split(", ")
                    temp[0] = temp[0].split(" ")
                    temp[0] = int(temp[0][len(temp[0])-1])
                    self.scan_list = []
                    for element in temp:
                        try:
                            self.scan_list.append(int(element))
                        except:
                            continue
                if currentline.startswith("# Internal Note: "):
                    temp = currentline.split(" ")
                    self.offset[0] = temp[len(temp)-3]
                    self.offset[1] = temp[len(temp)-2]
                    self.fit_data_type = int(temp[len(temp)-1])
            else:
                output += currentline
            currentline = fileInQuestion.readline()
        try:
            self.offset[0] = int(self.offset[0])
            self.offset[1] = int(self.offset[1])
        except ValueError:
            self.offset[0] = 0
            self.offset[1] = 0

        temp = Medium.ConvertToList(output)
        data = []
        data_index = 0
        for datum in range(len(temp)):
            while True:
                try:
                    temp[datum].remove(None)
                except:
                    break
            data.append(list(np.array(temp[datum]).astype(float)))

        # First the X-positions:
        x = data[0]
        # Now to adjust the separate the reading procedure depending on if the
        # file is a bulk file or just contains average data. In the latter
        # case, some features are effectively disabled, since OMEN isn't really
        # designed for single-run, single-scan analysis.
        if len(data) == 2:
            self.fit_function.append([copy.deepcopy(x), data[1]])
            self.fit_avg.append([copy.deepcopy(x), data[1]])
            if self.fit_data_type == 0:
                self.__slopes.append([copy.deepcopy(x), data[1]])
                self.AcquireAnalysisVariables()
            else:
                self.__heights.append([copy.deepcopy(x), data[1]])
                self.AcquireSlopes()
        else:
            data_index += 1
            # Now the data:
            for i in range(len(self.scan_list)):
                self.__slopes.append([copy.deepcopy(x), data[data_index]])
                data_index += 1

            for i in range(len(self.scan_list)):
                temp = [x[j] for j in range(len(data[data_index]))]
                self.__heights.append([temp, data[data_index]])
                data_index += 1

            if data_index == len(data):
                data_index = 1 + self.fit_data_type
            for i in range(len(self.scan_list)):
                temp = [x[j] for j in range(len(data[data_index]))]
                self.fit_function.append([temp, data[data_index]])
                data_index += 1

                data_index -= 1
            temp = [x[j] for j in range(len(data[data_index]))]
            self.fit_avg = [temp, data[data_index]]

    # wiping data, in case there are memory problems or leaks somewhere
    def clear(self):
        self.__slopes = []
        self.__heights = []
        self.__slopes_last = []
        self.__heights_last = []
        self.__slopes_initial = []
        self.__heights_initial = []
        self.slope_sigmas = []
        self.slope_avg = []
        self.height_sigmas = []
        self.height_avg = []
        self.residue = []
        self.residue_avg = []
        self.residue_sigmas = []
        self.residue_results = []
        self.height_results = []
        self.slope_results = []
        self.default_file = "C:\\Users\\Ben Sheff\\Documents\\Argonne\\"
        self.default_file += "Test Data\\Sheff_300mm_test0001_01_1.asc"
        self.scan_list = list(range(99))
        self.slope_deviation = []
        self.height_deviation = []
        self.fit_function = []
        self.fit_function_last = []
        self.fit_avg = []
        self.read_fit = ""
        self.read_fit = ""
        self.slope_deviation = []
        self.height_deviation = []
        self.filter_rigor = 0.0000000001
        self.filtered_list = list(range(99))
        self.radius = 0
        self.offset = [0, 0]

    # Returns the slope data that passed the filter plus offsets
    def GetSlopes(self):
        return [[[x + self.offset[i] for x in data] for
                 i, data in enumerate(self.__slopes[index])] for
                index in self.filtered_list]

    # NEEDS UPDATE: make output x values alongside, update Plottingtool and
    # GetHeightAverage accordingly
    # Returns the averaged slope data plus offsets
    def GetSlopeAverage(self):
        return [avg + self.offset[1] for avg in self.slope_avg]

    # Returns the height data that passed the filter plus offsets
    def GetHeights(self):
        return [[[x + self.offset[i] for x in data] for
                 i, data in enumerate(self.__heights[index])] for
                index in self.filtered_list]

    # Returns the averaged height data plus offsets
    def GetHeightAverage(self):
        return [avg + self.offset[1] for avg in self.height_avg]

    # Returns the residue data that passed the filter plus offsets
    def GetResidue(self):
        return [[[x + self.offset[i] for x in data] for
                 i, data in enumerate(self.residue[index])] for
                index in self.filtered_list]

    # Returns the averaged slope data plus offsets
    def GetResidueAverage(self):
        return [avg + self.offset[1] for avg in self.residue_avg]

    # Returns the fit to the average, plus offsets
    def GetFitAverage(self):
        return [[avg + self.offset[i] for avg in self.fit_avg[i]] for
                i in range(len(self.fit_avg))]

    # Returns the indices of the data that passed the filter
    def GetScanList(self):
        return [self.scan_list[index] for index in self.filtered_list]
