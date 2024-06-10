import numpy as np
import scipy.optimize as chisquarefit
import copy
import warnings

# Author: Ben Sheff
# Made for the Optics group in the XSD Division of the APS at Argonne over
# the summer of 2015.
#
# This class is intended to stitch together a series of data files given as
# python lists
#
# Known bugs: can get excessively slow fitting large segments to one another
# or many segments together


class Lightning:
    segments = -1  # Format is [RunNumber,
    #                          {0:x, 1:y, 2:chi^2/ndf of fit to left segment,
    #                           3:[x0, y0]}]
    unstitched_segments = -1  # Format is [RunNumber,
    #                           {0:x, 1:y, 2:IndexByCounter}]
    segment_counter = -1
    conglomerate = -1
    col_segments = -1
    segment_function = -1
    pixelsize = 0.5
    Total_RMS = 0
    last_fit_RMS = 0

    def __init__(self, segment=[]):
        self.unstitched_segments = []
        self.AddSegment(segment)
        if self.unstitched_segments != []:
            self.SetSegmentFunction(0)

    # Adds a new segment to the list of unmodified segments and updates all
    # other objects appropriately. It also checks if this is just updating or
    # replacing a previous segment based on range of x-values. Be wary using
    # speedflag, it removes the call to update other components of the obect.
    # It is for use only when adding a large number of segments at once, and
    # must be followed with the StitchAll method for proper functioning
    def AddSegment(self, segment=[], speedflag=False):
        if segment == []:
            return
        self.segment_counter += 1
        counter = self.segment_counter
        deleted = []

        # This loop checks if this segment is either contained by or contains
        # any already posessed segment. If it does, it removes the old ones
        for i in reversed(list(range(len(self.unstitched_segments)))):
            old_seg = self.unstitched_segments[i][0]
            length1 = len(old_seg) - 1
            length2 = len(segment[0]) - 1
            if ((old_seg[0] - segment[0][0] < 1 and
                 old_seg[length1] - segment[0][length2] > -1) or
                (segment[0][0] - old_seg[0] < 1 and
                 segment[0][length2] - old_seg[length1] > -1)):
                    # counter = self.unstitched_segments[i][2]
                    # deleted.append(counter)
                    # self.RemoveSegment(counter)
                    deleted.append(i)
                    del self.unstitched_segments[i]

        # for i in range(len(self.unstitched_segments)):
        #     if self.unstitched_segments[i][2] > counter:
        #         self.unstitched_segments[i][2] += 1

        temp = copy.deepcopy(segment)
        temp.append(counter)
        self.unstitched_segments.append(temp)

        if not speedflag:
            self.StitchAll()
        return deleted

    def GetMostRecent(self):
        maximum = output = -1
        for i, segment in enumerate(self.unstitched_segments):
            if segment[2] > maximum:
                maximum = segment[2]
                output = i
        return output

    # def GetSegment(self, index):
    #     output = -1
    #     for i, segment in enumerate(self.unstitched_segments):
    #         if segment[2] == index:
    #             output = i
    #     if output == -1:
    #         return []
    #     return self.segments[output]

    # This removes a segment from the unstitched list based on its index
    # given by the order in which it was added.
    def RemoveSegment(self, segment=-1):
        if segment == -1:
            segment = self.segment_counter

        for i in range(len(self.unstitched_segments)):
            if self.unstitched_segments[i][2] == segment:
                del self.unstitched_segments[i]
                self.segment_counter -= 1
            else:
                if self.unstitched_segments[i][2] > segment:
                    self.unstitched_segments[i][2] -= 1

        self.StitchAll()

    def SetPixelSize(self, pixel):
        self.pixelsize = pixel

    def __UpdateCollectedSegments(self):
        self.col_segments = [[], []]
        for segment in self.segments:
            for j in range(len(segment[0])):
                self.col_segments[0].append(segment[0][j])
                self.col_segments[1].append(segment[1][j])
        self.col_segments.append(0)

    def SetSegmentFunction(self, segment, conglomerate_flag=False):
        try:
            tmp = self.unstitched_segments[segment]
            self.segment_function = tmp
        except TypeError:
            self.segment_function = segment
            if not conglomerate_flag:
                self.AddSegment(segment)

    def __LagrangeInterpolate(self, x, x0, y0):
        output = 0
        for i in range(len(self.segment_function[1])):
            product = self.segment_function[1][i]
            for j in range(len(self.segment_function[1])):
                if i != j:
                    product *= x - x0 - self.segment_function[0][j]
                    product = product / (self.segment_function[0][i] -
                                         self.segment_function[0][j])
            output += product - y0

        return output

    def LinearInterpolate(self, x, x0, y0):
        if type(x0) == list or type(x0) == np.ndarray:
            return [self.LinearInterpolate(x[i], x0[i], y0[i])
                    for i in range(len(x0))]
        if type(x) == list or type(x) == np.ndarray:
            return [self.LinearInterpolate(_x, x0, y0) for _x in x]
        x = x - x0
        index = 1
        fit_seg = self.segment_function
        # if x < fit_seg[0][0]-1:
        #     return 9999999999999
        length = len(fit_seg[0])
        if fit_seg[0][0] > x:
            return fit_seg[1][0]
        while fit_seg[0][index] < x:
            index += 1
            if index == length:
                return fit_seg[1][length-1]
        return ((fit_seg[1][index-1] * (x - fit_seg[0][index]) -
                 fit_seg[1][index] * (x - fit_seg[0][index-1])) /
                (fit_seg[0][index-1] - fit_seg[0][index]) + y0)

    # Function being minimized in the fitting script, the chi square error of
    # the current fit. Note if the fit exceeds the boundary (x0 beyond
    # pixelsize), it returns an arbitrary large number
    def __FitError(self, params, X, Y, Err):
        x0 = params[0]
        y0 = params[1]
        if self.pixelsize < np.abs(x0):
            return 123456789098765432124.
        chi2 = 0.0
        for i in range(len(X)):
            x = X[i]
            y = self.LinearInterpolate(x, x0, y0)
            chi2 = chi2 + (Y[i]-y)**2 / (len(X) * Err[i]**2)
        return chi2

    # This is designed to fit the current segment_function to the indicated
    # function, assuming it's on the left of segment_function. startoverlap
    # is the index of the first point in the left segment in the range of
    # segment_function. It will ignore any points left of startoverlap. Pixel
    # is the allowed range for the fit to move the segment along x, a value
    # of -1 makes it just use the last defined value (default 0.5 mm)
    def FitSegments(self, left_segment, startoverlap=0, pixel=-1):
        if pixel == -1:
            pixel = self.pixelsize
        final_fit = [0.01, 0]
        self.pixelsize = pixel
        chisquare = 123456789098765432123.
        guess = final_fit.copy()
        # This is the biggest time sink in any of my code:
        for j in range(20):
            guess[0] += 0.1
            if guess[0] >= pixel:
                guess[0] = -pixel + 0.05
            current_range = [[], [], []]
            for i in range(startoverlap, len(left_segment[0])):
                current_range[0].append(left_segment[0][i])
                current_range[1].append(left_segment[1][i])
                current_range[2].append(1)
            fit = chisquarefit.fmin(self.__FitError, guess,
                                    args=tuple(current_range), disp=False)
            newchi = self.__FitError(fit, current_range[0],
                                     current_range[1], current_range[2])
            if newchi < chisquare:
                chisquare = newchi
                final_fit = fit
                guess = final_fit.copy()
        final_fit = list(final_fit)
        final_fit.append(chisquare)

        if abs(final_fit[0]) > self.pixelsize-0.005:
            warnings.warn("Fitting railed to pixel width")

        return final_fit

    def StitchSegments(self, segment=-1, addedsegment=-1):
        if segment == -1:
            segment = self.conglomerate
        if addedsegment != -1:
            self.SetSegmentFunction(addedsegment)
        try:
            temp = self.unstitched_segments[segment]
            segment_index = segment
            segment = copy.deepcopy(temp)
        except TypeError:
            self.AddSegment(segment)
            segment_index = len(self.unstitched_segments)-1
        temp = self.__Stitch(segment)
        self.unstitched_segments[segment_index][1] = temp[0][1]
        self.unstitched_segments[segment_index][0] = temp[0][0]
        output = self.unstitched_segments[segment_index]
        self.unstitched_segments.remove(self.segment_function)
        self.__SortSegments()
        return output

    def StitchAll(self):
        if self.unstitched_segments == []:
            return
        self.segments = []
        self.__SortSegments()
        self.conglomerate = self.unstitched_segments[0]
        for segment in range(len(self.unstitched_segments)):
            self.SetSegmentFunction(segment)
            if segment > 0:
                temp = self.__Stitch(self.conglomerate)
            else:
                temp = [copy.deepcopy(self.segment_function)]
                temp.append([0, 0, 0])
            self.conglomerate = temp[0]
            data = self.unstitched_segments[segment]
            self.segments.append([[x+temp[1][0] for x in data[0]],
                                  [y+temp[1][1] for y in data[1]],
                                  temp[1]])
        self.SetSegmentFunction(self.conglomerate, True)
        self.__UpdateCollectedSegments()
        temp = [1 for i in self.col_segments[0]]
        self.conglomerate.append(self.__FitError((0, 0), self.col_segments[0],
                                                 self.col_segments[1], temp))
        return self.conglomerate

    def __Stitch(self, segment):
        # First to find where along the left segment the overlap begins, then
        # calling the fitting function.
        startoverlap = 0
        while (startoverlap < len(segment[0]) and
               segment[0][startoverlap] < self.segment_function[0][0]):
            startoverlap += 1
        if startoverlap < len(segment[0]):
            x0, y0, chisquare = self.FitSegments(segment, startoverlap)
            startoverlap = 0
            while segment[0][startoverlap] < self.segment_function[0][0]+x0:
                startoverlap += 1
        else:
            x0 = y0 = chisquare = 0

        # Getting the list of x-values for the combined region of the segments
        endoverlap = 0
        while (endoverlap < len(self.segment_function[0]) and
               segment[0][len(segment[0])-1] >
               self.segment_function[0][endoverlap]):
            endoverlap += 1
        xvalues = [segment[0][i] for i in range(len(segment[0]))]
        for i in range(endoverlap, len(self.segment_function[0])):
            xvalues.append(self.segment_function[0][i] + x0)
        fitted_segment = self.LinearInterpolate(xvalues, x0, y0)

        # Finding the y-values along the combined region of the segments,
        # averaged where appropriate
        output = []
        for i in range(len(xvalues)):
            if i >= startoverlap and i < len(segment[0]):
                output.append((segment[1][i]+fitted_segment[i])/2)
            if i >= len(segment[0]):
                output.append(fitted_segment[i])
            if i < startoverlap:
                output.append(segment[1][i])
        return [[xvalues, output], [x0, y0, chisquare]]

    # This is a method to sort the segments by when they start
    def __SortSegments(self):
        self.unstitched_segments = list(self.unstitched_segments)
        for i in range(len(self.unstitched_segments)):
            self.unstitched_segments[i] = list(self.unstitched_segments[i])
            for j in range(len(self.unstitched_segments[i])-1):
                temp = list(self.unstitched_segments[i][j])
                self.unstitched_segments[i][j] = temp
        self.unstitched_segments.sort()


# import matplotlib.pyplot as mpl
# from LTPDataAnalyzer import LTPDataAnalyzer as LTP
# e1 = LTP("Segment4.asc")
# e2 = LTP("Segment5.asc")
# tmp = [list(e1.GetSlopes()[0][0]), list(e1.GetSlopes()[0][1])]
# tmp2 = [list(e2.GetSlopes()[0][0]), list(e2.GetSlopes()[0][1])]
# # [140 + 140*np.cos(((2.*float(i+1)-1)/802)* 3.14159265358979323846264338327950288419716) for i in range(401)] 
# # [[7.3242187e-05, 0.699985352, 1.40004883, 2.10013184, 2.80003906, 3.50000488, 4.19998047, 4.90005371, 5.60004883, 6.30007324, 7.00004883, 7.70003418, 8.39999512, 9.0999707, 9.8, 10.5000244, 11.2000244, 11.9000488, 12.5999512, 13.2999414, 13.9999805, 14.7000781, 15.4000195, 16.0999316, 16.8000391, 17.4999023, 18.2000293, 18.8999658, 19.5999902, 20.3000146, 21.0000977, 21.6999609, 22.4000781, 23.0999854, 23.8000537, 24.5000049, 25.2000293, 25.8999854, 26.6, 27.3000586, 28.0000977, 28.7000342, 29.3999951, 30.0999365, 30.8000098, 31.5000439, 32.1999316, 32.9000342, 33.5999902, 34.3000244, 34.9999951, 35.7000098, 36.3999756, 37.0999805, 37.7999414, 38.4999854, 39.2000049, 39.8999854, 40.5999414, 41.2999365, 41.9998975, 42.6999707, 43.3999121, 44.0999316, 44.7999854, 45.5000195, 46.1999707, 46.9000244, 47.6000146, 48.3000293, 49.0000098, 49.7000244, 50.4000098, 51.0999463, 51.7999609, 52.5000391, 53.1999854, 53.8999707, 54.6, 55.3000244, 56.0000244, 56.6999121, 57.4000098, 58.0999854, 58.7999951, 59.5001074, 60.1999268, 60.8999463, 61.5999609, 62.3000586, 63.0000195, 63.6999854, 64.3999854, 65.1000342, 65.8000391, 66.5000098, 67.2000146, 67.9000195, 68.6000488, 69.3000537, 69.9999756, 70.6999658, 71.3999414, 72.1000244, 72.7999951, 73.5000928, 74.1999951, 74.8999658, 75.6000195, 76.3000195, 76.9999609, 77.6999854, 78.4000098, 79.0999658, 79.8000146, 80.5000244, 81.200083, 81.9000195, 82.5998828, 83.2999707, 84.0000586, 84.7000146, 85.3999219, 86.0999756, 86.7999902, 87.4999805, 88.2000049, 88.9000342, 89.6000244, 90.2999756, 91.0000586, 91.6999854, 92.4000391, 93.0999365, 93.7999365, 94.4999463, 95.2000195, 95.9, 96.6000439, 97.2999951, 97.9999609, 98.7000049, 99.4000244, 100.100029, 100.799893, 101.500078, 102.2, 102.899961, 103.599932, 104.300029, 104.999951, 105.699971, 106.399932, 107.100054, 107.800054, 108.500059, 109.199976, 109.899941, 110.600029, 111.30001, 112.00002, 112.69999, 113.399941, 114.099995, 114.800083, 115.500029, 116.200029, 116.89998, 117.599976, 118.299956, 118.999976, 119.70002, 120.400112, 121.100015, 121.800015, 122.50001, 123.199932, 123.900044, 124.600024, 125.299912, 126.000024, 126.700049, 127.400029, 128.100029, 128.79998, 129.499971, 130.199956, 130.899985, 131.60002, 132.299971, 133.00002, 133.69999, 134.399937, 135.099951, 135.79998, 136.500039, 137.199922, 137.900015, 138.600024, 139.299995, 139.99998, 140.69998, 141.400029, 142.100005, 142.800049, 143.500044, 144.200005, 144.900049, 145.60001, 146.299961, 146.999951, 147.700034, 148.399951, 149.099971, 149.800068, 150.500098, 151.200034, 151.9, 152.599941, 153.3, 154.000015, 154.700024, 155.399961, 156.100005, 156.799976, 157.499946, 158.199946, 158.899951, 159.59998, 160.29998, 161.00002, 161.700054, 162.400034, 163.099951, 163.79999, 164.500039, 165.19998, 165.89999, 166.600024, 167.30002, 167.999951, 168.69999, 169.399976, 170.100039, 170.799971, 171.500098, 172.199912, 172.899941, 173.599932, 174.300015, 174.999976, 175.699971, 176.4, 177.099946, 177.800039, 178.499927, 179.200034, 179.900054, 180.600049, 181.299995, 182.000005, 182.69999, 183.40001, 184.099995, 184.799927, 185.499985, 186.199927, 186.899976, 187.600039, 188.3, 189.000024, 189.699961, 190.400034, 191.100015, 191.800024, 192.500059, 193.199922, 193.900068, 194.599995, 195.300039, 195.999937, 196.699995, 197.400088, 198.100005, 198.799976, 199.5, 200.199971, 200.900034, 201.600015, 202.29998, 203.000093, 203.700083, 204.399995, 205.099971, 205.800044, 206.500034, 207.200024, 207.900059, 208.600054, 209.3, 209.999971, 210.700039, 211.39999, 212.1, 212.800054, 213.500122, 214.200142, 214.900029, 215.600024, 216.29998, 216.99998, 217.69999, 218.399961, 219.099941, 219.799985, 220.500044, 221.200034, 221.900054, 222.60002, 223.299897, 223.999976, 224.700015, 225.399946, 226.099922, 226.79998, 227.499976, 228.199995, 228.899937, 229.599985, 230.30001, 230.999976, 231.699951, 232.399951, 233.10001, 233.800039, 234.500083, 235.20001, 235.899995, 236.59998, 237.299985, 237.999966, 238.699966, 239.400024, 240.10002, 240.800029, 241.500024, 242.199976, 242.90001, 243.60002, 244.300063, 245.000024, 245.699946, 246.39999, 247.100083, 247.800049, 248.500078, 249.200024, 249.900034, 250.599976, 251.300034, 252.000059, 252.69998, 253.400029, 254.099941, 254.799922, 255.499946, 256.2, 256.900078, 257.600029, 258.299956, 258.999976, 259.7, 260.400117, 261.100029, 261.800068, 262.500029, 263.200039, 263.90001, 264.6, 265.299985, 265.999941, 266.699927, 267.400034, 268.099971, 268.799985, 269.499971, 270.200039, 270.900024, 271.599956, 272.299976, 273.000049, 273.69998, 274.39998, 275.1, 275.800039, 276.499932, 277.199897, 277.90002, 278.6, 279.300005, 279.999961], [1.7573688, 1.88148105, 1.87000716, 1.7206037, 1.610551, 1.40030348, 1.53209198, 1.60247087, 1.04170299, 1.41977692, 1.14997804, 1.29122043, 1.26197004, 1.33469212, 1.27918088, 1.35376143, 1.16234076, 1.39206171, 1.41751444, 1.13882732, 0.992655993, 1.27586806, 1.31853163, 1.37202275, 1.08388174, 1.4058789, 1.48167145, 0.998635352, 1.14593792, 1.31303704, 1.22270012, 1.24629438, 1.20823646, 0.885673821, 0.85206002, 0.485460103, 0.975849152, 0.83347553, 0.756470919, 0.804225087, 0.750087559, 1.25534427, 1.02069438, 0.960173488, 0.823940873, 0.649973571, 0.408455521, 0.215257272, 0.654256046, 0.689647436, 0.453139186, 0.752350032, 0.968092144, 0.557939768, 0.735381544, 0.880017638, 0.841959774, 0.999847412, 0.969708145, 0.800993025, 0.792831957, 0.836707592, 0.413384467, 0.852140844, 1.07467031, 0.866200447, 1.00760448, 0.828950584, 1.13446403, 0.920257151, 1.03750134, 0.890683532, 1.16242158, 0.829192996, 1.13171673, 1.20363081, 1.02530015, 1.07232702, 1.23635566, 1.14278662, 0.963001549, 0.685041726, 1.08751786, 0.999766588, 0.966072083, 0.881391287, 0.714615345, 0.739340842, 0.906278372, 0.587190151, 0.775540292, 0.593977571, 0.758652627, 0.674052596, 0.783943713, 0.782812476, 0.844141424, 0.498873264, 0.510266423, 0.90304631, 0.735219955, 0.931246281, 0.798811316, 0.805921972, 0.834768355, 0.661285877, 0.676719069, 0.908056021, 0.771338582, 0.823860049, 0.869109333, 1.01043248, 0.96881932, 0.747421086, 0.704434276, 0.672598183, 0.676961482, 0.587917387, 0.993706465, 0.770853758, 0.950073183, 1.04501593, 1.21954882, 0.663386703, 0.742249727, 0.784590125, 0.781196415, 0.829354584, 0.387204528, 0.580402792, 0.453300804, 0.525861263, 0.587270975, 0.664356351, 0.915328205, 0.890521944, 1.19175279, 1.07087266, 1.06990302, 1.47456086, 1.42793787, 1.18900549, 1.40280843, 1.40482843, 1.39456654, 1.29485655, 1.03564286, 0.704757512, 0.789599895, 0.952012479, 1.01746225, 0.84026289, 0.59656322, 0.613935709, 0.0726412535, 0.654660046, 0.613693297, 0.898521364, 0.227296814, 0.703383863, 0.858928263, 0.618299067, 0.507034302, 0.399001658, 0.248951823, 0.121930644, -0.0554303639, -0.0299776457, -0.0151908286, 0.142292812, -0.0143020032, 0.118132934, 0.0251295082, 0.0310280751, -0.291696221, 0.224630341, -0.00105042965, 0.16782634, 0.309311122, 0.293231487, 0.404011399, 0.377104253, 0.366519153, 0.449260682, 0.503075004, 0.415970147, 0.206692234, 0.355772436, 0.304866999, 0.258567303, 0.0644802228, -0.0008080228, 0.277232617, 0.14843379, -0.0176956989, 0.401102513, 0.743461788, 0.373306543, 0.460977018, 0.606905937, 0.299210846, 0.257274449, 0.567797601, 0.763258338, 0.695222795, 0.626540899, 0.691344321, 0.724796474, 0.472774148, 0.54638499, 0.476087034, 0.556323707, 0.244265288, 0.734573543, 0.84066695, 0.699262917, 0.942396998, 0.935205579, 1.0504297, 1.11628354, 0.56496954, 0.888340294, 0.718170643, 0.754127681, 0.460734606, 0.557535708, 0.312704831, 0.395203948, 0.493459523, 0.569736898, 0.765682399, 1.02877462, 1.09268928, 1.01115978, 0.77408582, 0.760268629, 0.721806765, 0.310361564, 0.55438447, 0.767217636, 0.617087007, 0.972778678, 0.997423351, 0.696192443, 0.513417661, 0.636883557, 0.927690983, 1.19458091, 1.35432696, 1.0504297, 1.02182567, 1.31368351, 1.35998321, 1.3683058, 1.35214531, 1.4189688, 1.11062729, 1.35069096, 1.70145357, 1.62315619, 1.62727714, 1.31384504, 1.57338202, 1.44086623, 1.36095285, 1.15611899, 1.05390418, 1.23546684, 1.01229095, 1.20104504, 1.05859065, 1.07038784, 0.467764407, 0.544849753, 0.752834857, 0.569817662, 0.368135184, 0.312947243, 0.600280166, 0.604239464, 0.595997632, 0.580402792, 0.769803345, 0.794528842, 0.650135159, 0.77448988, 0.7824893, 0.702333391, 1.03693569, 0.781115651, 0.687546611, 0.691182733, 0.735058367, 0.797922492, 0.724392414, 0.755824506, 1.05228806, 0.864422798, 1.04735911, 0.920822799, 1.29574537, 1.24653673, 1.4125855, 1.28580666, 1.08121526, 1.34673166, 1.05972195, 1.12735343, 1.35675108, 1.09891105, 1.03168356, 0.968900144, 1.28265536, 0.878078401, 0.990635931, 1.09382045, 0.792589545, 1.10287035, 0.661285877, 0.805194736, 1.13729215, 1.17939007, 1.01269495, 1.17914772, 0.803417087, 0.893107593, 1.31327951, 1.25453615, 1.39036489, 1.27724159, 1.37565887, 1.36814415, 1.66274929, 1.67115271, 1.68270743, 1.57378602, 1.28241301, 1.30794656, 1.06917572, 1.39060724, 1.49524617, 1.56085765, 1.59180486, 1.6410135, 1.51956773, 1.44167423, 1.31311786, 1.46244049, 1.14755404, 1.30237114, 0.912096143, 0.923085272, 1.53249609, 1.00768518, 1.01665425, 0.734169543, 0.841232538, 0.986111045, 0.951042831, 0.826930523, 0.960577488, 1.20387316, 0.942558587, 1.19466174, 1.08493221, 1.03645086, 0.888340294, 0.775782704, 0.531274974, 1.38470864, 1.49960947, 1.38567829, 1.36159921, 1.32838953, 1.30317914, 1.48134816, 1.60117793, 1.33824742, 1.39553618, 1.32741988, 0.891168356, 1.20249951, 1.15232134, 1.09567893, 0.892865181, 0.910884082, 0.480127156, 0.469380438, 0.366276741, 0.506953478, 0.456856102, 0.397466421, 0.361671001, 0.222044662, 0.567312837, 0.699343741, 0.439241201, 0.577655494, 0.915489852, 0.755016506, 0.894723654]]
# # [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [500., 32., -6., 100., 16., 90., -10., 0., -100, 100]]
# #tmp = [[35.0020531, 35.5020775, 36.0020531, 36.5020775, 37.0020726, 37.5020238, 38.0020677, 38.5020775, 39.0019945, 39.5020335, 40.002097, 40.5020677, 41.0020531, 41.5021263, 42.0020921, 42.502058, 43.0020043, 43.5020628, 44.0021361, 44.5020628, 45.0020677, 45.5021361, 46.00218, 46.5020775, 47.0020628, 47.5021263, 48.0021166, 48.5020677, 49.0020677, 49.502058, 50.0020384, 50.5021361, 51.0021117, 51.5020677, 52.002097, 52.5020677], [-0.114609957, 0.13051185, -0.166872874, -0.441568315, -0.0589533448, -0.205657959, -0.219717562, -0.263641685, -0.117615797, 0.0465421118, -0.174338996, -0.148837805, -0.139529377, -0.106368124, -0.195089027, -0.446804285, -0.00678739138, -0.227765471, -0.216614753, -0.00727220532, -0.0376215428, 0.0621531121, 0.111119293, -0.0225923173, -0.154655561, 0.0834849179, -0.0904662311, 0.285943121, 0.301748037, 0.161636874, 0.192471027, 0.468426973, 0.427993506, -0.0925994143, -0.0344217718, 0.28749451]]
# #tmp[0].sort()
# a = Segment_Stitcher(tmp)
# plot = []
# xrang = []
# size = len(tmp[0])
# for x in range(0, 2*size-1):
#     if x % 2 == 0 :
#         xvalue = tmp[0][int(x/2)]
#     else:
#         xvalue = tmp[0][int(x/2)] + tmp[0][int(x/2)+1]
#         xvalue *= .5
#     y = a.LinearInterpolate(xvalue, 0, 0)
#     # print(y)
#     plot.append(y)
#     xrang.append(xvalue)

# print("ready with first plot")
# figure = mpl.figure("hi")
# fig = figure.add_subplot(111)
# fig.plot(xrang, plot)
# fig.plot(tmp[0], tmp[1], "p")
# # fig.set_yscale("log")
# # tmp2 = [tmp[0].copy(), tmp[1].copy()]
# #tmp2 = [[43.7520677, 44.2520433, 44.7520775, 45.2521361, 45.7520677, 46.2520335, 46.7520482, 47.2520824, 47.7520482, 48.2521654, 48.7520824, 49.2521361, 49.7521214, 50.252097, 50.7521117, 51.2520726, 51.752141, 52.2521263, 52.7520921, 53.2520482, 53.7520433, 54.2521019, 54.7520531, 55.2520531, 55.7521019, 56.2520482, 56.7521166, 57.2521166, 57.7520628, 58.2521019, 58.7519945, 59.2521312, 59.7520628, 60.252058, 60.7521068, 61.2520531], [-0.228347242, -0.213414982, -0.217584386, -0.0157079641, -0.20517315, -0.0917267501, -0.325116068, -0.183162615, 0.0273434911, -0.108210415, -0.350132436, -0.179671943, -0.12343356, -0.17637521, -0.0116355279, -0.118973278, 0.169393897, -0.224274814, -0.178896248, -0.166872874, 0.158340141, -0.0150292246, 0.131772354, -0.0793155208, -0.0951204449, 0.0681648031, 0.0587594174, 0.00775701879, -0.283131182, -0.249000311, -0.000969627348, -0.00145444099, 0.0400456116, 0.0136717455, 0.35585323, 0.365452558]]

# #tmp2[0] = [x for x in tmp2[0]]
# a.SetSegmentFunction(tmp2)
# startoverlap = 0
# while tmp[0][startoverlap] < tmp2[0][0]:
#     startoverlap += 1
# print(startoverlap)
# outfit = a.FitSegments(tmp, startoverlap)
# print(outfit)
# plot2 = []
# xrang2 = copy.deepcopy(tmp[0])
# for x in range(len(tmp2[0])):
#     xvalue = tmp2[0][x] + outfit[0]
#     # plot2.append(a.LinearInterpolate(xvalue, outfit[0], outfit[1]))
#     xrang2.append(xvalue)
# xrang2.sort()
# a.AddSegment(tmp)
# otherplot = a.StitchAll()
# fig.plot(xrang2, a.LinearInterpolate(xrang2, outfit[0], outfit[1]))
# fig.plot(np.array(tmp2[0]) + np.array([outfit[0] for i in tmp2[0]]),
#          np.array(tmp2[1]) + np.array([outfit[1] for i in tmp2[0]]), "p")
# fig.plot(otherplot[0], otherplot[1])

# mpl.show()

# print(a.LinearInterpolate(0.5, 0, 0))
