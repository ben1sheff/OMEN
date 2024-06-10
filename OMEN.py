from PyQt4123456 import QtGui, QtCore
import sys
from Raven import Raven
from Lightning import Lightning as SegS
import pyqtgraph as pg
import numpy as np
import copy


# Author: Ben Sheff
# Version 4.0
# Made for the Optics Group in Division XSD of the APS
# at Argonne National lab during the summer of 2015

# Working Title: Optical Metrology ENgine (OMEN)
#
# Most of the documentation is in the code itself, parameters are listed
# inline with their purposes, and each function and button has a header
# briefly discussing it. I have made an attempt to make each method and
# variable named in a somewhat intuitive way to its purpose.
# An important feature to note is the distinction between segmentation and
# full scan modes. The former refers to whenever you are taking only the
# average scan from each run, with a focus on combining runs (often called
# segments) and finding relevant statistics. The latter is focused on
# dealing with individual runs, filtering out bad scans, etc.
class OMEN(QtGui.QMainWindow):
    interest = 0  # the active trace, be it individual run or full stitched
    runs = []  # runs will hold our global Raven objects
    run_num = 0  # tells which Raven object is active
    segs = 0  # segs will be our global Lightning object
    show_average = False  # If true, the average of the data is plotted
    show_RMS = False  # If true, one sigma bands are added to the plot
    show_height = False  # If true, the height is plotted
    show_slope = True  # If true, the slope data is plotted
    show_fit = False  # If true, the fit to the data is plotted
    show_raw_data = True  # If true, all runs, after filtering  are plotted
    show_residue = False  # If true, The residue is plotted offset from data
    scan_number = 0  # Records the number of scans originally, for coloring
    filter_rigor = 1  # The level of filtering done, near zero is no filter
    currentfit = 9  # The current fit being conducted.
    #                 {9:circle, (0-5):polynomial of degree currentfit,
    #                  8:another mirror's trace}
    view_cursor = False  # If true, cursers will be added to the plot to
    #                      choose a region of interest
    offset_scale = 1.0  # A factor on the amount of offset the residue plot has
    slider_resolution = 200  # Resolution on the slider, higher is slower
    presentation_mode = True  # Toggles the background color
    init_menu = True
    disciples = []  # This holds the objects created in New(). Must be a list
    #            to hold multiple new objects
    view_mode = -4  # Default assumption is that we're analyzing one run
    show_runs = True  # sets whether the segments themselves are shown
    show_combo = True  # sets whether the combined segments run is shown
    use_fit_avg = False  # Toggles whether the average fit or by-run fits
    #                      are used
    x_label = "Position"
    x_units = "m"
    y_label = "Slope"
    y_units = "radians"
    title = "Measured Mirror Slope"

    __shortcuts = {"Exit":"Ctrl+Q", "Save":"Ctrl+S", "Save Avg":"",
                   "Save Eq":"Ctrl+Shift+S", "Load":"Ctrl+L", "Undo":"Ctrl+Z",
                   "Wipe":"Ctrl+Shift+Z"}
    __shortcuts.update({"Window Resize":"Ctrl+Shift+W", "Autosize X":"Ctrl+X", 
                        "Autosize Y":"Ctrl+Y", "Increase Offset":"Ctrl++",
                        "Decrease Offset":"Ctrl+=", "Change Y Label":"Ctrl+Shift+Y",
                        "Change X Label":"Ctrl+Shift+X", "Change Title":"Ctrl+Shift+T",
                        "Presentation Mode":""})
    __shortcuts.update({"Apply Filter":""})
    __shortcuts.update({"Add Cursor":"Ctrl+R", "Reset Cursor":"Ctrl+Shift+R",
                        "Set ROI":"Return"})
    __shortcuts.update({"Recalc Height":"Ctrl+Shift+H", "Recalc Slope":"Ctrl+Shift+G",
                        "Zero Pos":"Ctrl+0", "Fit Avg":"Ctrl+Alt+F", "Calibrate":"",
                        "Left Seg":"Ctrl+Left", "Right Seg":"Ctrl+Right"})

    __segments_are_slopes = True  # Denotes type of plots for segment mode,
    #                             once you enter segment mode, this is set
    #                             for the rest of the session

    # Pretty sure these don't need to be parameters, but I don't
    # know another way
    height_btn = 0  # To access the status of the height button to change it
    modes_btn = 0  # To access the mode change button to change its name
    slope_btn = 0  # To access the status of the slope button to change it
    horizontalSlider = 0  # To access the filter slider's position
    RMS_btn = 0  # To access the status of the button to show 1 sigma bands
    plotter = 0  # to access the plotting widget globally
    number_of_scans = 0  # Checks filter changed something before replotting
    leg = 0  # Allows the legend to be saved for live-updating
    cursor = 0  # To access the cursor to define a region of interest
    profile_fit = []  # to access the RMS and P-V of the profile

    # Unfortunately GUIs get rather verbose due to the need to clarify what
    # every button does. I've made some attempt to improve legibility, but
    # at a certain point, there's just a lot of tedium to wade in. Please
    # read pyqt_first_look for some introduction on how to read this
    def __init__(self, parent=None):
        # Initializing our analysis object
        super(OMEN, self).__init__()
        self.runs = []
        self.runs.append(Raven())
        try:
            self.runs[self.run_num].GetSlopes()[0][0]
            self.init_menu = False
        except:
            # label = QtGui.QLabel()
            # pixmap = QtGui.QPixmap(os.getcwd() + "/optics picture.png")
            # label.setPixmap(pixmap)
            # self.setCentralWidget(label)

            name = QtGui.QFileDialog.getOpenFileName(self, "Load:")
            if not self.runs[self.run_num].Load(name):
                raise FileNotFoundError("Bad File")
        self.interest = self.runs[self.run_num]
        self.interest.FitCircle(0)

        self.scan_number = len(self.interest.scan_list_initial)
        # Now to make the window where the rest of the program will take place
        self.setWindowTitle("OMEN")
        self.setWindowIcon(QtGui.QIcon("optics picture.png"))
        self.resize(800, 500)
        mainMenu = self.menuBar()

        # Now we need to have a PlotWidget to be the center of our window.
        # Most of the work is done in the Plot method, but here some basics
        # are set up, like blue for the legend color, and the legend itself.
        pg.setConfigOption("foreground", "#07E")  # 07E is the hex RGB values
        #                                           for the legend font color
        self.plotter = pg.PlotWidget()
        self.AutoX()
        # fontsize = {"color": "#999", "font-size": "16px"}
        # self.plotter.setLabel("bottom", "position", units="m", **fontsize)
        self.setCentralWidget(self.plotter)
        self.leg = self.plotter.addLegend()
        self.Presentation()
        if (self.interest.fit_data_type == 1):
            self.PlotHeight()
        else:
            self.PlotSlope()
        self.Plot()

        # First, the File menu, for basic, necessary actions
        file_menu = mainMenu.addMenu("&File")
        self.FileMenu(file_menu)

        # Next a plotting menu
        plot_menu = mainMenu.addMenu("&Plot")
        self.PlotMenu(plot_menu)

        # Fitting menu
        fit_menu = mainMenu.addMenu("&Fit")
        self.FitMenu(fit_menu)

        # Now a view menu to make view manipulation easier
        view_menu = mainMenu.addMenu("&View")
        self.ViewMenu(view_menu)

        # Now for the filtering options, assumes filtering on plotted data
        filters = QtGui.QToolBar("Filters")
        self.Filters(filters)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, filters)

        # Dealing with cursors to define a region of interest
        cursor_menu = mainMenu.addMenu("&ROI")
        self.CursorMenu(cursor_menu)

        # Now a view menu to make view manipulation easier
        adv_menu = mainMenu.addMenu("&Advanced")
        self.AdvancedMenu(adv_menu)

        # This button is to change between modes for segments vs. scans
        self.modes_btn = QtGui.QAction("Segments Mode", self)
        self.modes_btn.triggered.connect(lambda: self.SegmentMode(plot_menu))
        self.modes_btn.triggered.connect(self.Plot)
        mainMenu.addAction(self.modes_btn)

        # # This button is for debugging purposes
        # test_btn = QtGui.QAction("test", self)
        # test_btn.triggered.connect(self.test)
        # mainMenu.addAction(test_btn)

    def SegmentMode(self, plot_menu):
        if self.view_mode <= -2:
            self.__segments_are_slopes = self.show_slope
            self.segs = SegS()

        self.view_mode = (self.view_mode + 5) % 2
        plot_menu.clear()

        if self.view_mode == 0:
            self.modes_btn.setText("Segments Mode")
            self.PlotMenu(plot_menu)
            self.interest = self.runs[self.run_num]
            self.ReFit()
            self.AutoY()
            self.scan_number = len(self.interest.scan_list_initial)
        if self.view_mode == 1:
            self.modes_btn.setText("Single Run Mode")
            self.SegPlotMenu(plot_menu)
            self.__UpdateSegments()
            self.AutoX()
            self.AutoY()

    # This menu covers basic things a user will want to do: saving, loading,
    # undo-ing, etc.
    def FileMenu(self, file_menu):
        save_btn = file_menu.addAction("Save", self.Save)
        save_btn.setShortcut(self.__shortcuts["Save"])
        save_avg_btn = file_menu.addAction("Save Average", self.SaveAvg)
        save_avg_btn.setShortcut(self.__shortcuts["Save Avg"])
        save_eq_btn = file_menu.addAction("Save Fit Data", self.SaveEquation)
        save_eq_btn.setShortcut(self.__shortcuts["Save Eq"])
        load_btn = file_menu.addAction("Load", self.Load)
        load_btn.triggered.connect(self.Plot)
        load_btn.setShortcut(self.__shortcuts["Load"])
        file_menu.addAction("New", self.New)
        undo_btn = file_menu.addAction("Undo", self.Undo)
        undo_btn.setShortcut(self.__shortcuts["Undo"])
        undo_btn.triggered.connect(self.Plot)
        wipe_btn = file_menu.addAction("Wipe clean", self.Wipe)
        wipe_btn.setShortcut(self.__shortcuts["Wipe"])
        wipe_btn.triggered.connect(self.Plot)
        exit_btn = file_menu.addAction("Exit", self.Exit)
        exit_btn.setShortcut(self.__shortcuts["Exit"])

    # This defines a menu for different plotting options, mainly for deciding
    # what you want to plot. A different method is used when in segment mode
    def PlotMenu(self, plot_menu):
        # A checkable menu for the option plotting height or slope
        plot_types = QtGui.QActionGroup(self, exclusive=True)
        self.slope_btn = plot_types.addAction(QtGui.QAction("Slope", self,
                                                            checkable=True))
        self.slope_btn.setShortcut("G")
        self.slope_btn.triggered.connect(self.PlotSlope)
        self.slope_btn.triggered.connect(self.Plot)
        self.slope_btn.setChecked(True)
        plot_menu.addAction(self.slope_btn)

        self.height_btn = plot_types.addAction(QtGui.QAction("Height", self,
                                                             checkable=True))
        self.height_btn.setShortcut("H")
        self.height_btn.triggered.connect(self.PlotHeight)
        self.height_btn.triggered.connect(self.Plot)
        plot_menu.addAction(self.height_btn)
        if self.show_height:
            self.height_btn.setChecked(True)

        # self.slope_btn = QtGui.QAction("Slope", self, checkable=True)
        # self.slope_btn.setShortcut("Ctrl+G")
        # self.slope_btn.triggered.connect(self.PlotSlope)
        # self.slope_btn.triggered.connect(self.Plot)
        # plot_menu.addAction(self.slope_btn)
        # if not(self.slope_btn.isChecked()):
        #     self.slope_btn.setChecked(True)
        # self.height_btn = QtGui.QAction("Height", self, checkable=True)
        # self.height_btn.setShortcut("Ctrl+H")
        # self.height_btn.triggered.connect(self.PlotHeight)
        # self.height_btn.triggered.connect(self.Plot)
        # plot_menu.addAction(self.height_btn)

        # Checkable button, default on, to choose whether to show all raw runs
        raw_btn = QtGui.QAction("Show Raw Data", self, checkable=True)
        raw_btn.triggered.connect(self.PlotRawData)
        raw_btn.triggered.connect(self.Plot)
        raw_btn.setShortcut("Ctrl+D")
        raw_btn.setChecked(self.show_raw_data)
        plot_menu.addAction(raw_btn)

        # Checkable button, default off, to show average, brings RMS along with
        avg_btn = QtGui.QAction("Show Average", self, checkable=True)
        avg_btn.setChecked(self.show_average)
        avg_btn.triggered.connect(self.PlotAverage)
        avg_btn.triggered.connect(self.Plot)
        avg_btn.setShortcut("Ctrl+A")
        plot_menu.addAction(avg_btn)

        # Checkable button, default off, to show 1 sigma bands, can go with avg
        self.RMS_btn = QtGui.QAction("Show Standard Deviation", self,
                                     checkable=True)
        self.RMS_btn.setChecked(self.show_RMS)
        self.RMS_btn.triggered.connect(self.PlotRMS)
        self.RMS_btn.triggered.connect(self.Plot)
        self.RMS_btn.setShortcut("Ctrl+Shift+A")
        plot_menu.addAction(self.RMS_btn)

        # Checkable button, default off, to show the fit
        plot_fit_btn = QtGui.QAction("Show Fit", self, checkable=True)
        plot_fit_btn.setChecked(self.show_fit)
        plot_fit_btn.triggered.connect(self.PlotFit)
        plot_fit_btn.triggered.connect(self.Plot)
        plot_fit_btn.setShortcut("Ctrl+F")
        plot_menu.addAction(plot_fit_btn)

        # Checkable button, default off, to show the residue after fitting
        residue_btn = QtGui.QAction("Show Fit Residue", self, checkable=True)
        residue_btn.setChecked(self.show_residue)
        residue_btn.triggered.connect(self.PlotFitResidue)
        residue_btn.triggered.connect(self.Plot)
        residue_btn.setShortcut("Ctrl+Shift+F")
        plot_menu.addAction(residue_btn)

    def SegPlotMenu(self, plot_menu):
        # Checkable button, default on, to show the individual runs
        runs_btn = QtGui.QAction("Show Runs", self, checkable=True)
        runs_btn.setChecked(self.show_runs)
        runs_btn.triggered.connect(self.PlotRuns)
        runs_btn.triggered.connect(self.Plot)
        runs_btn.setShortcut("R")
        plot_menu.addAction(runs_btn)
        if not runs_btn.isChecked():
            runs_btn.setChecked(True)

        # Checkable button, default off, to show the conglomerate
        combo_btn = QtGui.QAction("Show Stitched Trace", self, checkable=True)
        combo_btn.setChecked(self.show_combo)
        combo_btn.triggered.connect(self.PlotCombo)
        combo_btn.triggered.connect(self.Plot)
        combo_btn.setShortcut("T")
        plot_menu.addAction(combo_btn)

        # Checkable button, default off, to show the fit
        plot_fit_btn = QtGui.QAction("Show Fit", self, checkable=True)
        plot_fit_btn.setChecked(self.show_fit)
        plot_fit_btn.triggered.connect(self.PlotFit)
        plot_fit_btn.triggered.connect(self.Plot)
        plot_fit_btn.setShortcut("Ctrl+F")
        plot_menu.addAction(plot_fit_btn)

        # Checkable button, default off, to show the residue after fitting
        residue_btn = QtGui.QAction("Show Fit Residue", self, checkable=True)
        residue_btn.setChecked(self.show_residue)
        residue_btn.triggered.connect(self.PlotFitResidue)
        residue_btn.triggered.connect(self.Plot)
        residue_btn.setShortcut("Ctrl+Shift+F")
        plot_menu.addAction(residue_btn)

    # This defines a menu for different fitting options, mainly for deciding
    # the curve to fit and for applying it
    def FitMenu(self, fit_menu):
        # First a checkable, exlusive menu for the functions. Default is circle
        functs = QtGui.QActionGroup(self, exclusive=True)
        # This adds the list of polynomials to choose from, it will give the
        # options of polynomials up to degree n-1 where n is number in range()
        polynomials = fit_menu.addMenu("&Polynomial degree:")
        for degree in range(6):
            poly_btn = functs.addAction(QtGui.QAction("Degree " + str(degree),
                                                      self, checkable=True))
            poly_btn.setShortcut("Ctrl+"+str(degree))
            poly_btn.triggered.connect(self.PolynomialInterface)
            poly_btn.triggered.connect(self.Plot)
            poly_btn.uniqueId = str(degree)
            polynomials.addAction(poly_btn)
        # This is a button in the above menu to choose a circle for the fit
        circle_btn = functs.addAction(QtGui.QAction("circle",
                                                    self, checkable=True,
                                                    checked=True))
        circle_btn.setShortcut("Ctrl+9")
        circle_btn.triggered.connect(self.Circle)
        circle_btn.triggered.connect(self.Plot)
        fit_menu.addAction(circle_btn)
        if not(circle_btn.isChecked()):
            circle_btn.setChecked(True)  # Is the default fit

        compare_btn = functs.addAction(QtGui.QAction("Other Mirror",
                                                     self, checkable=True,
                                                     checked=False))
        compare_btn.setShortcut("Ctrl+8")
        compare_btn.triggered.connect(self.CompareMirrors)
        compare_btn.triggered.connect(self.Plot)
        fit_menu.addAction(compare_btn)

        # Button to subtract the fit from each of the runs then recalc. average
        # (note doesn't directly subtract the fit from the average)
        subtract_btn = fit_menu.addAction("Subtract Fit", self.SubtractFit)
        subtract_btn.triggered.connect(self.Plot)
        subtract_btn.setShortcut("Ctrl+-")

        # Button to open a window with the current average fit function
        return_btn = fit_menu.addAction("Return Fit", self.ReturnFit)
        return_btn.setShortcut("Ctrl+shift+-")

    # This defines a menu for manipulating the view of the plot
    def ViewMenu(self, view_menu):
        # First a button to re-size the whole window
        autowindow_btn = view_menu.addAction("Autosize window", self.AutoX)
        autowindow_btn.triggered.connect(self.AutoY)
        autowindow_btn.triggered.connect(self.Plot)
        autowindow_btn.setShortcut(self.__shortcuts["Window Resize"])

        # Now a button to resize the X-axis
        autox_btn = view_menu.addAction("Autosize X", self.AutoX)
        autox_btn.triggered.connect(self.Plot)
        autox_btn.setShortcut(self.__shortcuts["Autosize X"])

        # Now a button to resize the Y-axis
        autoy_btn = view_menu.addAction("Autosize Y", self.AutoY)
        autoy_btn.triggered.connect(self.Plot)
        autoy_btn.setShortcut(self.__shortcuts["Autosize Y"])

        # Now a button to increase the offset of the residue data
        offset_up_btn = view_menu.addAction("Increase Residue Offset",
                                            self.UpOffSet)
        offset_up_btn.triggered.connect(self.Plot)
        offset_up_btn.setShortcut(self.__shortcuts["Increase Offset"])
        # Now a button to decrease the offset of the residue data
        offset_down_btn = view_menu.addAction("Decrease Residue Offset",
                                              self.DownOffSet)
        offset_down_btn.triggered.connect(self.Plot)
        offset_down_btn.setShortcut(self.__shortcuts["Decrease Offset"])

        view_menu.addAction("Add Vertical Space", self.y_padding)

        # Button to change the y-axis label
        y_label_btn = view_menu.addAction("Change Y-axis Label",
                                          lambda: self.SetAxisLabel("Y"))
        y_label_btn.triggered.connect(self.Plot)
        y_label_btn.setShortcut(self.__shortcuts["Change Y Label"])
        # Button to change the x-axis label
        x_label_btn = view_menu.addAction("Change X-axis Label",
                                          lambda: self.SetAxisLabel("X"))
        x_label_btn.triggered.connect(self.Plot)
        x_label_btn.setShortcut(self.__shortcuts["Change X Label"])
        # Button to change the plot title
        title_btn = view_menu.addAction("Change Title",
                                        self.SetTitle)
        title_btn.triggered.connect(self.Plot)
        title_btn.setShortcut(self.__shortcuts["Change Title"])

        background_btn = QtGui.QAction("Presentation Mode", self,
                                       checkable=True)
        background_btn.triggered.connect(self.Presentation)
        background_btn.triggered.connect(self.Plot)
        view_menu.addAction(background_btn)
        background_btn.setShortcut(self.__shortcuts["Presentation Mode"])

    # This defines a toolbar to deal with adding filters to your data.
    def Filters(self, filters):
        # This is a slider to adjust the strength of the filter
        self.horizontalSlider = QtGui.QSlider()
        info = "Move the slider to test filters of varying strengths"
        self.horizontalSlider.setMaximum(self.slider_resolution)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.setSliderPosition(0)
        self.horizontalSlider.setGeometry(QtCore.QRect(30, 30, 600, 20))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setInvertedControls(False)
        self.horizontalSlider.setObjectName("Filter Tuning")
        self.horizontalSlider.setToolTip(info)
        self.horizontalSlider.valueChanged.connect(self.FilterRigor)
        filters.addWidget(self.horizontalSlider)

        # Now for the filter itself, this button applies the filter.
        filter_btn = QtGui.QPushButton("Apply Filter")
        info = "Apply filter on current plotted data with rigor set on slider"
        filter_btn.setToolTip(info)
        filter_btn.released.connect(self.Filter)
        filter_btn.released.connect(self.Plot)
        filter_btn.setShortcut(self.__shortcuts["Apply Filter"])
        filters.addWidget(filter_btn)

    # This defines a menu to work with cursors to define a region of interest.
    def CursorMenu(self, cursor_menu):
        # Checkable button, default off, to toggle the visibility of the cursor
        view_cursor_btn = QtGui.QAction("Add Cursor",
                                        self, checkable=True, checked=False)
        view_cursor_btn.triggered.connect(self.AddCursor)
        view_cursor_btn.triggered.connect(self.Plot)
        view_cursor_btn.setShortcut(self.__shortcuts["Add Cursor"])

        # Button to reset cursor to a small region in middle of the data range
        cursor_menu.addAction(view_cursor_btn)
        cursor_res_btn = cursor_menu.addAction("Reset Cursor", self.CursorRes)
        cursor_res_btn.triggered.connect(self.Plot)
        cursor_res_btn.setShortcut(self.__shortcuts["Reset Cursor"])

        # Button to apply the chosen region of interest to the data, cutting
        # off outside data.
        cursor_apply_btn = cursor_menu.addAction("Set ROI", self.SetROI)
        cursor_apply_btn.triggered.connect(self.Plot)
        cursor_apply_btn.setShortcut(self.__shortcuts["Set ROI"])

        # Defining the cursor itself
        self.cursor = pg.LinearRegionItem([0.1, 0.2])
        self.cursor.setZValue(-10)

    # This defines a menu to do more specific analysis things that don't fit
    # the theme of any other menu.
    def AdvancedMenu(self, adv_menu):
        # Button to Recalculate the height from the current selection of slopes
        recalc_btn = adv_menu.addAction("Recalculate Height", self.ReCalcH)
        recalc_btn.triggered.connect(self.Plot)
        recalc_btn.setShortcut(self.__shortcuts["Recalc Height"])

        # Button to Recalculate the slope from the current selection of heights
        slope_recalc_btn = adv_menu.addAction("Recalculate Slope",
                                              self.ReCalcS)
        slope_recalc_btn.triggered.connect(self.Plot)
        slope_recalc_btn.setShortcut(self.__shortcuts["Recalc Slope"])

        # Button to shift over the data so the x-range is centered on zero
        zero_x_btn = adv_menu.addAction("Zero the Position", self.ZeroX)
        zero_x_btn.triggered.connect(self.Plot)
        zero_x_btn.setShortcut(self.__shortcuts["Zero Pos"])

        # Button to toggle whether you are using the average fit or not
        avg_fit_btn = QtGui.QAction("Use Average Fit", self, checkable=True)
        avg_fit_btn.setChecked(self.use_fit_avg)
        avg_fit_btn.triggered.connect(self.UseFitAvg)
        avg_fit_btn.triggered.connect(self.Plot)
        avg_fit_btn.setShortcut(self.__shortcuts["Fit Avg"])
        adv_menu.addAction(avg_fit_btn)

        # Button to apply a calibration curve
        cal_btn = adv_menu.addAction("Apply Calibration", self.Calibrate)
        cal_btn.triggered.connect(self.Plot)
        cal_btn.setShortcut(self.__shortcuts["Calibrate"])


        # Buttons to shift segment choice right or left:
        seg_left_btn = adv_menu.addAction("Select segment to left",
                                          lambda: self.ShiftSegment(-1))
        seg_left_btn.triggered.connect(self.Plot)
        seg_left_btn.setShortcut(self.__shortcuts["Left Seg"])
        seg_right_btn = adv_menu.addAction("Select segment to right",
                                           lambda: self.ShiftSegment(1))
        seg_right_btn.triggered.connect(self.Plot)
        seg_right_btn.setShortcut(self.__shortcuts["Right Seg"])

    # This is the method that manages the actual plotting. Most of the plotting
    # itself is delegated to two other methods to segregate the two main modes
    # of operation in this tool.
    # It works by clearing the plot and re-doing it all each time, which is
    # why it only is called when you actually make a change. You'll notice
    # most buttons are set up to trigger this method when they are hit.
    def Plot(self):
        # First we need to clear out the window to make a new plot
        self.plotter.clear()
        self.leg.items = []  # The legend has to be cleared or it keeps growing
        # Now to do the plotting itself:
        if not(self.view_mode <= -2):
            self.PlotSegments()
        if self.view_mode <= 0:
            self.PlotRun()
        # Finally, adjusting the labels to reflect any changes, specifically
        # whether or not we're in presentation mode, and if we're shoing slope
        # or height data.
        if self.presentation_mode:
            color = "#000"  # This part assigns color by hex values
            label_size = "28px"  # font size for axis labels
            title_size = "40px"  # font size for the title
        else:
            color = "#999"
            label_size = "16px"
            title_size = "28px"
        fontsize = {"color": color, "size": title_size}
        self.plotter.setTitle(self.title, **fontsize)
        fontsize = {"color": color, "font-size": label_size}
        self.plotter.setLabel("left", self.y_label, units=self.y_units, **fontsize)
        self.plotter.setLabel("bottom", self.x_label, units=self.x_units, **fontsize)

    # This is the method that does the actual plotting when dealing with
    # individual segments or runs. When combining segments, see PlotSegments.
    # It's rather large, but it integrates all the plotting options selected
    # in other methods.
    def PlotRun(self):
        # First to parse out the data itself, be it height or slope. We also
        # need the scale to let pyqtgraph know the units to use.
        all_data = []
        if self.show_height:
            all_data.append((0.000000001, np.array(self.interest.GetHeights()),
                             np.array(self.interest.GetHeightAverage()),
                             np.array(self.interest.height_sigmas),
                             self.interest.height_results))
            rad_units = " km"
        if self.show_slope:
            all_data.append((0.000001, np.array(self.interest.GetSlopes()),
                             np.array(self.interest.GetSlopeAverage()),
                             np.array(self.interest.slope_sigmas),
                             self.interest.slope_results))
            rad_units = " m^2"

        # Adding the cursor, and confirming that the data exists
        if self.view_cursor:
            self.plotter.addItem(self.cursor)

        try:
            data_results = all_data[0][4]
        except:
            return

        # In order to get the residue and plot it, it too has to be read out,
        # and an offset is generated based on a rough estimate of the needed
        # space given what is being plotted.
        residue_results = self.interest.residue_results
        if self.show_residue:
            residue = np.array(self.interest.GetResidue())
            residue_avg = np.array(self.interest.residue_avg)
            residue_sigmas = np.array(self.interest.residue_sigmas)
            offset = data_results[2]
            if self.show_raw_data:
                offset += data_results[1] + data_results[2]
            else:
                if self.show_RMS:
                    offset += data_results[1]
            offset = offset*self.offset_scale + data_results[0]
        for scale, data, data_avg, data_sigmas, data_results in all_data:
            # Plotting all the runs together
            if(self.show_raw_data):
                for i in range(len(data)):
                    # The following line sets color for each raw plot based on
                    # pyqtgraphs int-to-color alg., takes (color, num colors)
                    pen = pg.mkPen((2*self.scan_number -
                                    self.interest.GetScanList()[i],
                                    3*self.scan_number))
                    self.plotter.plot(data[i][0]*0.001,
                                      data[i][1]*scale, pen=pen)
                    if self.show_residue:
                        tmp = offset*scale
                        self.plotter.plot(residue[i][0]*0.001,
                                          residue[i][1]*scale + tmp, pen=pen)

            # # Plotting the fit to the data, along with setting its legend entry
            if(self.show_fit):
                pen = pg.mkPen(0.6, width=3)  # color for fit, note grey-scaled
                name = "Fit to Data | Residue RMS = {0:.3g} | "
                name += "Residue P-V = {1:.3g}"
                if self.currentfit == 9 or self.currentfit == 2:
                    temp = " | R = {0:.4g}" + rad_units
                    name += temp.format(self.interest.radius)
            # Uncomment below, and comment matching text above for legend to
            # list the fit:
            #
            # if(self.show_fit):
            #     pen = pg.mkPen(0.6, width=3)  # color for fit, note grey-scaled
            #     name = "Fit to Data | "# Residue RMS = {0:.3g} | "
            #     name += self.interest.read_fit  #"Residue P-V = {1:.3g}"
            #     if self.currentfit == 9 or self.currentfit == 2:
            #         temp = " | R = {0:.4g}" + rad_units
            #         name += temp.format(self.interest.radius)
                if self.currentfit == 8:
                    if self.view_mode > -2:
                        name = "Unstitched Data | "
                    else:
                        name = "Stitched Data | "
                    name += "RMS = {0:.3g} | P-V = {1:.3g}"
                    name = name.format(self.profile_fit.slope_results[1],
                                       self.profile_fit.slope_results[2])
                else:
                    name = name.format(residue_results[1], residue_results[2])
                self.plotter.plot(np.array(self.interest.GetFitAverage()[0])*0.001,
                                  np.array(self.interest.GetFitAverage()[1])*scale,
                                  pen=pen, name=self.__NM(name, 5))

            # Plotting the average of the data, and again a legend entry
            if(self.show_average):
                pen = pg.mkPen((200, 0, 0), width=3)  # color for average, RGB
                pen2 = pg.mkPen((100, 0, 0), width=3)  # color for residue avg
                name = "Average Data | RMS = {0:.3g} | P-V = {1:.3g}"
                if self.currentfit == 8:
                    if self.view_mode <= -2:
                        name = "Unstitched Data | "
                    else:
                        name = "Stitched Data | "
                    name += "RMS = {0:.3g} | P-V = {1:.3g}"
                name = name.format(data_results[1], data_results[2])
                self.plotter.plot(data[0][0]*0.001, data_avg*scale,
                                  pen=pen, name=self.__NM(name, 5))
                if self.show_residue:
                    tmp = residue_avg*scale + offset*scale
                    name = "Average Residue | RMS = {0:.3g} | P-V = {1:.3g}"
                    name = name.format(residue_results[1], residue_results[2])
                    self.plotter.plot(residue[0][0]*0.001, tmp,
                                      pen=pen2, name=self.__NM(name, 5))

            # Plotting the 1 sigma bands, and adding a legend entry.
            if(self.show_RMS):
                name = "One Sigma Band"
                pen = pg.mkPen((170, 0, 0), width=2)  # color of 1 sigma band
                p2 = pg.mkPen((70, 0, 0), width=2)  # color of band of residue
                self.plotter.plot(data[0][0]*0.001, (data_avg+data_sigmas)*scale,
                                  pen=pen, name=self.__NM(name, 5))
                self.plotter.plot(data[0][0]*0.001,
                                  (data_avg-data_sigmas)*scale, pen=pen)
                if self.show_residue:
                    temp = residue_sigmas*scale + (residue_avg + offset)*scale
                    self.plotter.plot(residue[0][0]*0.001,
                                      temp, pen=p2, name=self.__NM(name, 5))
                    self.plotter.plot(residue[0][0]*0.001,
                                      temp - 2*residue_sigmas*scale, pen=p2)

    def PlotSegments(self):
        # First to parse out the data itself, be it height or slope. We also
        # need the scale to let pyqtgraph know the units to use.
        all_data = []

        if self.__segments_are_slopes:
            for seg in self.segs.segments:
                all_data.append((0.000001, [np.asarray(x) for x in seg],
                                 self.interest.slope_results))
            units = "radians"
            rad_units = " m^2"
        else:
            for seg in self.segs.segments:
                all_data.append((0.000000001, [np.asarray(x) for x in seg],
                                 self.interest.height_results))
            units = " " + chr(181) + "m"
            rad_units = " km"

        # In order to get the residue and plot it, it too has to be read out,
        # and an offset is generated based on a rough estimate of the needed
        # space given what is being plotted.
        residue_results = self.interest.residue_results
        if self.show_residue and self.view_mode == 1:
            residue = np.array(self.interest.GetResidue())
            residue_avg = np.array(self.interest.residue_avg)
            offset = all_data[0][2][2]*self.offset_scale + all_data[0][2][0]

        # Now we need to clear out the window to make a new plot
        if self.view_cursor:
            self.plotter.addItem(self.cursor)

        try:
            all_data[0][1]
        except:
            return

        index = -1
        for scale, data, data_results in all_data:
            index += 1
            # Plotting all the runs together
            if(self.show_runs):
                # The following line sets color for each raw plot based on
                # pyqtgraphs int-to-color alg., takes (color, num colors)
                pen = pg.mkPen(2*len(self.segs.segments)-index,
                               3*len(self.segs.segments), width=2)
                name = "Segment " + str(index)
                if index > 0:
                    name += ", fitted with X^2/ndf: "
                    name += "{0:.3g}".format(float(data[2][2])) + units
                self.plotter.plot(data[0]*0.001, data[1]*scale,
                                  pen=pen, name=self.__NM(name, 5))

        # Plotting the fit to the data, along with setting its legend entry
        if self.show_fit and self.view_mode == 1:
            pen = pg.mkPen(0.6, width=3)  # color for fit, note grey-scaled
            name = "Fit to Data | Residue RMS = {0:.3g} | "
            name += "Residue P-V = {1:.3g}"
            if self.currentfit == 9 or self.currentfit == 2:
                temp = " | R = {0:.4g}" + rad_units
                name += temp.format(self.interest.radius)
            name = name.format(residue_results[1], residue_results[2])
            self.plotter.plot(np.array(self.interest.GetFitAverage()[0])*0.001,
                              np.array(self.interest.GetFitAverage()[1])*scale,
                              pen=pen, name=self.__NM(name, 5))

        if self.show_residue and self.view_mode == 1:
            p2 = pg.mkPen((150, 0, 0), width=3)  # color for residue avg
            tmp = residue_avg*scale + offset*scale
            name = "Average Residue | RMS = {0:.3g} | P-V = {1:.3g}"
            name = name.format(residue_results[1], residue_results[2])
            self.plotter.plot(residue[0][0]*0.001, tmp,
                              pen=p2, name=self.__NM(name, 5))

        if self.show_combo:
            pen = pg.mkPen("#D33", style=QtCore.Qt.DashLine, width=2)
            temp = [np.array(x) for x in self.segs.conglomerate]
            name = "Stitched Trace, total X^2/ndf: "
            name += "{0:.3g}".format(float(temp[2])) + units
            self.plotter.plot(temp[0]*0.001, temp[1]*scale, pen=pen, name=self.__NM(name, 5))

    # -------------------------------------------------------------------------
    # Section I: File Menu functions
    # -------------------------------------------------------------------------

    # Saves the current equation, and statistics on it, to a file from the user
    def SaveEquation(self):
        tmp = "Equation.txt"
        name = QtGui.QFileDialog.getSaveFileName(self, "Save As:", tmp, ".txt")
        if name == "":
            return
        fileInQuestion = open(name, "w")
        fileInQuestion.write("Fitted Equation: " + str(self.interest.read_fit))

        fileInQuestion.write("\n\nRMS of Average: " +
                             str(self.interest.slope_results[1]))
        fileInQuestion.write("\n\nPeak to Valley of Average: " +
                             str(self.interest.slope_results[2]))
        fileInQuestion.write("\n\nRMS of Residue: " +
                             str(self.interest.residue_results[1]))
        fileInQuestion.write("\n\nPeak to Valley of Residue: " +
                             str(self.interest.residue_results[2]))
        if self.currentfit == 9:
            fileInQuestion.write("\n\nRadius of Circle: " +
                                 str(self.interest.radius))

    # Saves the current data, so you can re-load your current progress later
    def Save(self):
        tmp = self.interest.default_file.format(0)
        name = QtGui.QFileDialog.getSaveFileName(self, "Save As:", tmp, ".asc")
        if name == "":
            return
        if self.view_mode == 1:
            for i, segment in enumerate(self.runs):
                tmp = name.split(".")
                tmp.insert(1, "{0:04d}.")
                segment.Save((tmp[0] + tmp[1] + tmp[2]).format(i))
            self.interest.SaveAvg(name)
        else:
            self.interest.Save(name)

    # Saves the current Average, so you can do further analysis elsewhere :(
    def SaveAvg(self):
        tmp = self.interest.default_file.format(0)
        name = QtGui.QFileDialog.getSaveFileName(self, "Save As:", tmp, ".asc")
        if name == "":
            return
        self.interest.SaveAvg(name)
        if self.view_mode == 1:
            for i, segment in enumerate(self.runs):
                tmp = name.split(".")
                tmp.insert(1, "{0:04d}.")
                segment.SaveAvg((tmp[0] + tmp[1] + tmp[2]).format(i))

    # Loads your data from a previous session
    def Load(self):
        name = QtGui.QFileDialog.getOpenFileName(self, "Load:")
        if name == "":
            return

        self.runs.append(Raven([]))
        self.run_num = len(self.runs) - 1
        try:
            load_success = self.runs[self.run_num].Load(name)
            if not load_success:
                raise IndexError("Bad File")
        except IndexError:
            del self.runs[self.run_num]
            self.run_num -= 1
            QtGui.QMessageBox.critical(self, "Bad File",
                                       "This file cannot be read")
            return

        if self.view_mode <= -2:
            del self.runs[0]
            self.run_num -= 1
            self.show_height = False
            self.show_slope = True
            if self.runs[self.run_num].fit_data_type == 1:
                self.show_height = True
                self.show_slope = False
            self.interest = self.runs[self.run_num]
            self.ReFit()
            self.scan_number = len(self.interest.scan_list_initial)
        else:
            temp = self.runs[self.run_num]
            if ((self.__segments_are_slopes ^ temp.fit_data_type == 0) and
                    load_success >= 2):
                del self.runs[self.run_num]
                self.run_num -= 1
                QtGui.QMessageBox.critical(self, "Bad File",
                                           "Inconsistent data type (height" +
                                           " vs. slope)")
                return
            self.__UpdateSegments(True)

        self.slope_btn.setChecked(self.show_slope)
        self.height_btn.setChecked(self.show_height)
        self.AutoX()
        self.AutoY()
        self.horizontalSlider.setSliderPosition(0)

    # Opens a new instance of this analysis engine.
    def New(self):
        self.disciples.append(OMEN(self))
        self.disciples[len(self.disciples)-1].show()

    # Undoes the last significant change. You can only undo one change, since
    # it currently works by saving the entirety of the data, not the changes
    def Undo(self):
        self.interest.Undo()
        self.ReFit()
        self.AutoX()

    # Undoes everything you've done since loading the file(s), restoring the
    # data to its original state. This can be undone with the Undo method
    def Wipe(self):
        self.interest.ReturnToStartPoint()

    # Quits the program
    def Exit(self):
        for element in self.runs:
            element.clear()
        self.close()

    # -------------------------------------------------------------------------
    # Section II: Plot Menu functions
    # -------------------------------------------------------------------------

    # Method to add the residue to the fit to the shown plot, vertically
    # offset. Note that while this is on, live-updating the filter (in filter
    # rigor) will likely be much slower.
    def PlotFitResidue(self):
        self.show_residue = not(self.show_residue)

    # Sets the show_height parameter to make the plot show slopes, not heights
    def PlotSlope(self):
        self.show_slope = True
        self.show_height = False
        self.title = "Measured Mirror Slope"
        self.y_label = "Slope"
        self.y_units = "radians"

        self.plotter.getViewBox().enableAutoRange(y=True)
        self.ReFit()

    # Sets the show_height parameter to make the plot show heights, not slopes
    def PlotHeight(self):
        self.show_slope = False
        self.show_height = True
        self.title = "Mirror Height Error from a Plane"
        self.y_label = "Height"
        self.y_units="meters"

        self.plotter.getViewBox().enableAutoRange(y=True)
        self.ReFit()

    # Toggles the show_fit parameter changing if current fit is shown
    def PlotFit(self):
        self.show_fit = not(self.show_fit)

    # Toggles the show_average parameter changing showing average of the data,
    # and if RMS has the same visiblity as the Average, it toggles RMS as well
    def PlotAverage(self):
        self.show_average = not(self.show_average)
        if self.show_average ^ self.show_RMS:
            self.show_RMS = self.show_average
            self.RMS_btn.setChecked(not(self.RMS_btn.isChecked()))

    # Toggles the show_RMS parameter changing if 1 sigma bands are shown
    def PlotRMS(self):
        self.show_RMS = not(self.show_RMS)

    # Toggles the show_raw_data parameter changing if all the data are shown
    def PlotRawData(self):
        self.show_raw_data = not(self.show_raw_data)

    # Toggles the show_runs parameter changing showing all runs overlaying the
    # stitched version.
    def PlotRuns(self):
        self.show_runs = not(self.show_runs)

    # Toggles the show_combo parameter changing showing stitched data.
    def PlotCombo(self):
        self.show_combo = not(self.show_combo)

    # -------------------------------------------------------------------------
    # SECTION III: Fit Menu functions
    # -------------------------------------------------------------------------

    # Opens a notification window with the fitted function in string form.
    def ReturnFit(self):
        QtGui.QMessageBox.information(self, "Fitted Function",
                                      self.interest.read_fit)

    # Subtracts the fit from the data, making the data just the residue, to
    # allow for iterative fitting.
    def SubtractFit(self):
        if self.view_mode == 1:
            # f = SegS(copy.deepcopy(self.interest.fit_avg))
            for segment in self.runs:
                # if self.__segments_are_slopes:
                #     x = segment.GetSlopes()[0][0]
                # else:
                #     x = segments.GetHeights()[0][0]
                # segment.fit_function = [[funct[0], f.LinearInterpolate(funct[0],
                #                          -segment.offset[0], -segment.offset[1])]
                #                         for funct in segment.fit_function]
                # segment.fit_avg = [x - segment.offset[0],
                #                    f.LinearInterpolate(x, 0, -segment.offset[1])]
                segment.SubtractFit()
            self.__UpdateAllSegments()
        else:
            self.interest.SubtractFit()
        self.__UpdateSegments()

    # Fits a circle to the current data and to the average of the data.
    def Circle(self):
        self.currentfit = 9
        self.ReFit()

    def CompareMirrors(self):
        name = QtGui.QFileDialog.getOpenFileName(self, "Comparison:")
        if name == "":
            return

        comparison_run = Raven()
        try:
            load_success = comparison_run.Load(name)
            if not load_success:
                raise IndexError("Bad File")
        except IndexError:
            QtGui.QMessageBox.critical(self, "Bad File",
                                       "This file cannot be read")
            return

        if ((self.show_slope ^ comparison_run.fit_data_type == 0) and
                load_success >= 2):
            QtGui.QMessageBox.critical(self, "Bad File",
                                       "Inconsistent data type (height" +
                                       " vs. slope)")
            return
        self.currentfit = 8
        self.profile_fit = comparison_run

        # Now that we have the file and are sure it's in the right form:
        # Fit the comparison run to the active one:
        comparison_data = []
        original_data = []
        if self.show_slope:
            original_data.append(self.interest.GetSlopes()[0][0])
            original_data.append(self.interest.GetSlopeAverage())
            comparison_data.append(comparison_run.GetSlopes()[0][0])
            comparison_data.append(comparison_run.GetSlopeAverage())
        else:
            original_data.append(self.interest.GetHeights()[0][0])
            original_data.append(self.interest.GetHeightAverage())
            comparison_data.append(comparison_run.GetHeights()[0][0])
            comparison_data.append(comparison_run.GetHeightAverage())
        # Since stitching will only adjust data on the right side, we must
        # force the new data to be on the right
        while comparison_data[0][0] <= original_data[0][0]:
            comparison_data[0].pop(0)
            comparison_data[1].pop(0)
        fitter = SegS(original_data)
        comparison_data.append(1)
        fitter.unstitched_segments.append(comparison_data)
        fitter.StitchAll()
        fitter.SetSegmentFunction(fitter.segments[fitter.GetMostRecent()])


        # Then we have to make this the function the Raven object
        # thinks it's fitted on, to retain nice features like residue finding
        if self.__segments_are_slopes:
            x = self.interest.GetSlopes()
        else:
            x = self.interest.GetHeights()
        self.interest.fit_function = [[funct[0],
                                       fitter.LinearInterpolate(funct[0], 0, -self.interest.offset[1])]
                                      for funct in x]
        self.interest.fit_avg = [[i - self.interest.offset[0] for i in x[0][0]],
                                  fitter.LinearInterpolate(x[0][0], 0, -self.interest.offset[1])]
        self.ReFit()


        self.AutoX()
        self.AutoY()
        self.horizontalSlider.setSliderPosition(0)

    # Since the button is bad at handling passing parameters, this function is
    # called when the fit button is hit. The signal triggering this function
    # includes a tag clarified with the button, that I've named degree, from
    # which this interface gets the degree of the polynomial to be fit.
    def PolynomialInterface(self):
        button = QtCore.QObject.sender(self)
        degree = button.uniqueId
        self.currentfit = int(degree)
        return self.ReFit()

    # Fits a polynomial to the current data and to the average of the data.
    # Now obsolete
    # def Polynomial(self, degree):
    #     if self.show_height:
    #         self.interest.FitPolynomial(degree, 1)
    #     else:
    #         self.interest.FitPolynomial(degree, 0)

    # -------------------------------------------------------------------------
    # SECTION IV: View Menu functions
    # -------------------------------------------------------------------------

    # Adjusts the X-axis to be a reasonable scale, padding indicates how much
    # of a fraction of the total width should be added ad additional space
    # at the edges
    def AutoX(self, padding=0):
        if self.show_height:
            x = self.interest.GetHeights()[0][0]
        else:
            x = self.interest.GetSlopes()[0][0]
        xmax = x[len(x)-1]
        xmin = x[0]
        self.plotter.setXRange(*(0.001*xmin, 0.001*xmax), padding=padding)

    # Adjusts the Y-axis to be a reasonable scale, padding is set to default,
    # which works pretty well here. If you want to specify, add
    # ", padding=" some number to the parameters in "enableAutoRange"
    def AutoY(self):
        self.plotter.getViewBox().enableAutoRange(y=True)

    # Increases the offset of the residue plotted alongside regular data by
    # by increasing a factor it's multiplied by 0.5, with 1 being the default
    def UpOffSet(self):
        self.offset_scale += 0.5

    # Decreases the offset of the residue plotted alongside regular data by
    # by decreasing a factor it's multiplied by 0.5, with 1 being the default
    def DownOffSet(self):
        self.offset_scale -= 0.5

    # This is a temporary method to add space for the legend, will be updated
    def y_padding(self):
        if self.show_slope:
            stats = np.array(self.interest.slope_results) * 0.000001
        else:
            stats = np.array(self.interest.height_results) * 0.000000001
        if self.show_residue:
            stats[2] *= 0.5 + abs(self.offset_scale)
        self.plotter.setYRange(*(stats[0]-2*stats[2],
                                 stats[0]+3*stats[2]))

    # Method to allow the user to choose his or her labels for the axes
    def SetAxisLabel(self, which_axis):
        new_label, flag = QtGui.QInputDialog.getText(self, "Enter label", which_axis+"-axis label. Units should be in parenthesis without scaling prefix",
                                                     QtGui.QLineEdit.Normal,
                                                     "Label(units)")
        if (not flag) or (new_label == "Label(units)" or new_label == ""):
            return
        if not new_label.endswith(")"):
            if which_axis == "Y":
                self.y_label = new_label
            else:
                self.x_label = new_label
            return

        new_label = new_label.split(")")
        new_label.remove("")

        for char_seg in np.delete(new_label, 0):
            new_label[0] += ")" + char_seg
        new_label = new_label[0].split("(")
        if len(new_label) == 1:
            new_label.append("")
        temp_units = new_label.pop()
        if temp_units != "":
            if which_axis == "Y":
                self.y_units = temp_units
            else:
                self.x_units = temp_units

        for char_seg in np.delete(new_label, 0):
            new_label[0] += "(" + char_seg
        if new_label[0] != "":
            if which_axis == "Y":
                self.y_label = new_label[0]
            else:
                self.x_label = new_label[0]

    # Method to allow the user to choose his or her labels for the axes
    def SetTitle(self):
        new_title, flag = QtGui.QInputDialog.getText(self, "Enter Title",
                                                     "Graph Title",
                                                     QtGui.QLineEdit.Normal,
                                                     self.title)
        if (not flag) or new_title == "":
            return
        self.title = new_title

    # Toggles the background between white and black
    def Presentation(self):
        black = "#000"
        white = "#FFF"
        grey = "#999"
        self.presentation_mode = not(self.presentation_mode)
        font = QtGui.QFont()  # This is the font used for the tick mark text
        #                       the SetPixelSize method below sets the size
        if self.presentation_mode:
            font.setPixelSize(15)
            self.plotter.setBackground(white)
            axis = self.plotter.getAxis("left")
            axis.setStyle(**{"tickLength":8})
            axis.tickFont = font
            axis.setPen(pg.mkPen(black))
            axis = self.plotter.getAxis("bottom")
            axis.setPen(pg.mkPen(black))
            axis.setStyle(**{"tickLength":8})
            axis.setPen(pg.mkPen(black))
            axis.tickFont = font
        else:
            font.setPixelSize(12)
            self.plotter.setBackground(black)
            axis = self.plotter.getAxis("left")
            axis.setPen(pg.mkPen(grey))
            axis.setStyle(**{"tickLength":-6})
            axis.tickFont = font
            axis = self.plotter.getAxis("bottom")
            axis.setPen(pg.mkPen(grey))
            axis.setStyle(**{"tickLength":-6})
            axis.tickFont = font

    # -------------------------------------------------------------------------
    # SECTION V: Filter functions
    # -------------------------------------------------------------------------

    # Updates the filter, called by a signal from the cursor moving. To change
    # to only trigger when released (for speed), change:
    # self.horizontalSlider.valueChanged.connect(self.FilterRigor) above to:
    # self.horizontalSlider.sliderReleased.connect(self.FilterRigor)
    def FilterRigor(self):
        value = self.horizontalSlider.value()
        value = float(value)/self.slider_resolution
        if value >= 1:
            value -= 0.000000001
        if value <= 0:
            value += 0.000000001
        self.filter_rigor = 1./(1. - value) - 1
        number_of_scans = len(self.interest.filtered_list)
        if self.show_height:
            self.interest.FilterHeights(self.filter_rigor)
        else:
            self.interest.FilterSlopes(self.filter_rigor)
        if number_of_scans != len(self.interest.filtered_list):
            self.Plot()

    # Applies the selected filter and strength to the data, allowing iterative
    # filtering
    def Filter(self):
        self.interest.ApproveFilter()
        self.ReFit()
        self.__UpdateSegments()
        self.horizontalSlider.setSliderPosition(0)
        self.FilterRigor()

    # -------------------------------------------------------------------------
    # SECTION VI: Cursor functions
    # -------------------------------------------------------------------------
    #  NEEDS UPDATE: Cannot account for when segments are shortened to nothing
    # or to be completely overlapping another one.
    # Calls for the data to be cut down to the region of interest
    def SetROI(self):
        if not(self.view_cursor):
            return
        self.interest.SetROI(list(1000*np.array(self.cursor.getRegion())))
        if self.view_mode == 1:
            for segment in self.runs:
                segment.SetROI(list(1000*np.array(self.cursor.getRegion())))
            self.__UpdateAllSegments()
            self.__UpdateSegments
        self.ReFit()
        self.AutoX(0.01)

    # Resets the region of interest to be the middle third of the data span
    def CursorRes(self):
        temp = len(self.interest.GetSlopes()[0][0])-1
        xmax = self.interest.GetSlopes()[0][0][temp]
        xmin = self.interest.GetSlopes()[0][0][0]
        xmax = 2*(xmax*0.001-xmin*0.001) / 3 + xmin*0.001
        xmin = (xmax-xmin*0.001) / 2 + xmin*0.001
        self.cursor.setRegion([xmin, xmax])

    # Toggles cursor visiblity and useability.
    def AddCursor(self):
        self.CursorRes()
        self.view_cursor = not(self.view_cursor)

    # -------------------------------------------------------------------------
    # SECTION VII: Advanced functions
    # -------------------------------------------------------------------------

    # NEEDS UPDATE: only sets active Raven to zero
    # Sets the middle of the x-range to be zero, shifting everything over
    def ZeroX(self):
        temp = len(self.interest.GetSlopes()[0][0])-1
        xmax = self.interest.GetSlopes()[0][0][temp]
        xmin = self.interest.GetSlopes()[0][0][0]
        offset = (xmax + xmin) / 2000
        self.interest.ZeroX(offset * 1000)
        self.AutoX()
        self.cursor.setRegion([self.cursor.getRegion()[0]-offset,
                               self.cursor.getRegion()[1]-offset])

    # NEEDS UPDATE: only works on active Raven
    # Adds a calibration curve, applying it to the active object
    def Calibrate(self):
        name = QtGui.QFileDialog.getOpenFileName(self, "Calibration Curve:")
        if name == "":
            return

        calibration_curve = Raven()
        try:
            load_success = calibration_curve.Load(name)
            if not load_success:
                raise IndexError("Bad File")
        except IndexError:
            QtGui.QMessageBox.critical(self, "Bad File",
                                       "This file cannot be read")
            return

        if ((self.show_slope ^ calibration_curve.fit_data_type == 0) and
                load_success >= 2):
            QtGui.QMessageBox.critical(self, "Bad File",
                                       "Inconsistent data type (height" +
                                       " vs. slope)")
            return
        self.interest.AddCalibration(calibration_curve)
        self.ReFit()

    # Re-calculates the height data based on the current available slope data
    def ReCalcH(self):
        self.interest.AcquireAnalysisVariables()
        self.ReFit()

    # Re-calculates the height data based on the current available slope data
    def ReCalcS(self):
        self.interest.AcquireSlopes()
        self.ReFit()

    # Changes the fitting to use the average fit instead of the fits on each
    # run
    def UseFitAvg(self):
        self.use_fit_avg = not self.use_fit_avg
        self.ReFit()

    # Shifts the segment of interest
    def ShiftSegment(self, direction):
        if self.view_mode < 0:
            QtGui.QMessageBox.warning(self, "Mode Error",
                                      "Please initiate segment mode first")
            return
        if self.view_mode == 1:
            QtGui.QMessageBox.warning(self, "Mode Error",
                                      "Please enter Single Run Mode first.")
            return
        self.__UpdateSegments()
        self.run_num += direction
        self.run_num = self.run_num % len(self.runs)
        self.interest = self.runs[self.run_num]
        self.scan_number = len(self.interest.scan_list_initial)
        self.ReFit()

    # -------------------------------------------------------------------------
    # SECTION VIII: Utility functions
    # -------------------------------------------------------------------------

    # Used to change the a name to use a given font size, given in pixels
    def __NM(self, name, size):
        return "<font size='" + str(size) + "'>" + name + "</font>"

    # Used for resetting after a change, re-does the current fit choice
    # particularly useful for any change the backend is blind to, like changing
    # showing slope vs. height
    # Note this includes a loop to update all the segments' fits to be part
    # of the total mirrors fit while in segment mode. This is more robust, but
    # significantly slower, especially when dealing with many segments. If
    # speed is a concern, move the code in "if self.view_mode == 1:" to
    # SubtractFit and to Load, and you should be fine
    def ReFit(self):
        self.interest.UseFitAverage(self.use_fit_avg)
        data_type = int(self.show_height)%2
        if self.currentfit == 9:
            self.interest.FitCircle(data_type)
        if self.currentfit >= 0 and self.currentfit <= 5:
            self.interest.FitPolynomial(self.currentfit, data_type)
        if self.currentfit == 8:
            # We're fitting another mirror shape on, so  don't do anything
            self.interest.Reset()

        if self.view_mode == 1:
            f = SegS(copy.deepcopy(self.interest.fit_avg))
            for segment in self.runs:
                if self.__segments_are_slopes:
                    x = segment.GetSlopes()[0][0]
                else:
                    x = segment.GetHeights()[0][0]
                segment.fit_function = [[funct[0], f.LinearInterpolate(funct[0],
                                         -segment.offset[0], -segment.offset[1])]
                                        for funct in segment.fit_function]
                segment.fit_avg = [[i - segment.offset[0] for i in x],
                                   f.LinearInterpolate(x, 0, -segment.offset[1])]

    # This handles updating the Lightning
    # object to reflect the current data
    def __UpdateSegments(self, remove_redundencies=False):
        if self.view_mode < 0:
            self.scan_number = len(self.interest.scan_list_initial)
            self.ReFit()
            return

        active = self.runs[self.run_num]
        if self.__segments_are_slopes:
            bad_segs = self.segs.AddSegment([[x-active.offset[0] for x in
                                              active.GetSlopes()[0][0]],
                                             active.slope_avg])
        else:
            bad_segs = self.segs.AddSegment([[x-active.offset[0] for x in
                                              active.GetHeights()[0][0]],
                                             active.height_avg])

        if remove_redundencies:
            self.RemoveSegments(bad_segs)
            self.run_num = self.segs.GetMostRecent()
            self.runs.insert(self.run_num, self.runs.pop())

        # Updating all the offsets
        for i, segment in enumerate(self.runs):
            offset = self.segs.segments[i]
            segment.SetOffset(offset[2][0], offset[2][1])

        if self.view_mode == 1:
            self.interest = Raven([self.segs.conglomerate[0],
                                             self.segs.conglomerate[1]])
            self.interest.fit_data_type = (1-int(self.__segments_are_slopes))%2
        if self.view_mode <= 0:
            self.interest = self.runs[self.run_num]

        self.scan_number = len(self.interest.scan_list_initial)

        if self.__segments_are_slopes:
            self.PlotSlope()
        else:
            self.PlotHeight()

    def RemoveSegments(self, delete_indices=[]):
        for index in reversed(delete_indices):
            del self.runs[index]
            if self.run_num >= index:
                self.run_num -= 1

    def __UpdateAllSegments(self):
        self.segs = SegS()
        for segment in self.runs:
            if self.__segments_are_slopes:
                self.segs.AddSegment([[x-segment.offset[0] for x in
                                       segment.GetSlopes()[0][0]],
                                      [y-segment.offset[1] for y in
                                       segment.GetSlopeAverage()]],
                                     True)
            else:
                self.segs.AddSegment([[x-segment.offset[0] for x in
                                       segment.GetHeights()[0][0]],
                                      [y-segment.offset[1] for y in
                                       segment.GetHeightAverage()]],
                                     True)
        self.segs.StitchAll()
        for i, segment in enumerate(self.runs):
            offset = self.segs.segments[i]
            segment.SetOffset(offset[2][0], offset[2][1])

    def test(self):
        for i, seg in enumerate(self.runs):
            print(seg.GetSlopes()[0][0][0])
            print(self.segs.segments[i])

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = OMEN()
    main.show()
    sys.exit(app.exec_())
