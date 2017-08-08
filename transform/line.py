from collections import deque
import numpy as np


class Line:
    def __init__(self, side):
        self.side = side
        # Was the line found in the previous frame?
        self.found = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0

    def found_search(self, x, y):
        """
        This function is applied when the lane lines have been detected in the previous frame.
        It uses a sliding window to search for lane pixels in close proximity (+/- 25 pixels in the x direction)
        around the previous detected polynomial.
        """
        xvals = []
        yvals = []
        if self.found:
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i, j])
                xval = (np.mean(self.fit0)) * yval ** 2 + (np.mean(self.fit1)) * yval + (np.mean(self.fit2))
                x_idx = np.where((((xval - 25) < x) & (x < (xval + 25)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) == 0:
            self.found = False  # If no lane pixels were detected then perform blind search
        return xvals, yvals, self.found

    def blind_search(self, x, y, image):
        """
        This function is applied in the first few frames and/or if the lane was not successfully detected
        in the previous frame. It uses a slinding window approach to detect peaks in a histogram of the
        binary thresholded image. Pixels in close proimity to the detected peaks are considered to belong
        to the lane lines.
        """
        xvals = []
        yvals = []
        if not self.found:
            i = 720
            j = 630
            while j >= 0:
                histogram = np.sum(image[j:i, :], axis=0)
                if self.side == "right":
                    peak = np.argmax(histogram[640:]) + 640
                else:
                    peak = np.argmax(histogram[:640])
                x_idx = np.where((((peak - 25) < x) & (x < (peak + 25)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) > 0:
            self.found = True
        else:
            yvals = self.Y
            xvals = self.X
        return xvals, yvals, self.found

    @staticmethod
    def radius_of_curvature(xvals, yvals):
        # meters per pixel in y dimension
        ym_per_pix = 30. / 720
        # meteres per pixel in x dimension
        xm_per_pix = 3.7 / 700
        fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * np.max(yvals) + fit_cr[1]) ** 2) ** 1.5) \
                   / np.absolute(2 * fit_cr[0])
        return curverad

    @staticmethod
    def sort_vals(xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    @staticmethod
    def get_intercepts(polynomial):
        bottom = polynomial[0] * 720 ** 2 + polynomial[1] * 720 + polynomial[2]
        top = polynomial[0] * 0 ** 2 + polynomial[1] * 0 + polynomial[2]
        return bottom, top
