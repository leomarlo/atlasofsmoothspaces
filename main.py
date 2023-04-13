import numpy as np
import time

MIDI_MAX : int = 127

INITIAL_SIGMOID_CALIBRATION = {
                    "XMIN": 0,
                    "XMAX": 0,
                    "YMIN": 0,
                    "YMAX": MIDI_MAX,
                    "BETA": 0.02}

INITIAL_SMOOTHNESS_VALUES = {
                "raw": 0,
                "calibration": INITIAL_SIGMOID_CALIBRATION,
                "calibratedSignal": 0,
                "isCalibrated" : False}

def sigmoid(x:float, YMAX:float, YMIN:float, XMAX:float, XMIN:float, BETA=0.02):
    """ Implements a simple sigmoid function
    """
    MU = (XMAX - XMIN) / 2
    ALPHA = np.log((1-BETA) / BETA) / MU
    return YMIN + (YMAX-  YMIN) / (1 + np.exp((-1)* ALPHA * (x - MU)))


class SmoothnessBase():
    """It creates instances that can deal with smoothness.
    In particular it takes new values and calculates the smoothness of them.
    It is agnostic to the source of the data.
    """

    
    def __init__(self, 
        cacheLength: int,
        derivativeDegree: int,
        alpha: float or list[float]):

        self.cacheLength = cacheLength
        self.derivativeDegree = derivativeDegree
        self.smoothness = np.zeros(self.derivativeDegree + 1)
        self.smoothnessMeasures = {
            "overall": INITIAL_SMOOTHNESS_VALUES,
            "relative": INITIAL_SMOOTHNESS_VALUES, 
            "secondOrder": INITIAL_SMOOTHNESS_VALUES}
        self.derivatives = np.zeros((self.derivativeDegree + 1, self.cacheLength), float)  
        self.deltas = np.zeros(cacheLength, float)
        if isinstance(alpha, list):
            if len(alpha) != self.derivativeDegree + 1:
                raise Exception("There need to be as many values of alpha as there are degrees of derivatives!")
            self.__alphaIsList = True
            self.alpha = np.array(alpha)
        else:
            self.__alphaIsList = False
            self.alpha = alpha
        
        ## lastUpdateTime is in milliseconds
        self.lastUpdateTime = time.time()*1000.0
                


    def _addNewDerivative(self, newValue: float, newDelta: float):
        """add a value to the derivatives array

        Args:
            value (float): new value
        """
        newArray = np.zeros(self.derivativeDegree + 1, float)
        oldValue = 0
        newValue *= newDelta
        for i, value in enumerate(self.derivatives[:,self.cacheLength - 1]):
            newValue = (newValue - oldValue) / newDelta
            oldValue = value
            newArray[i] = newValue

        self.derivatives = np.delete(np.insert(self.derivatives, self.cacheLength, newArray, axis=1),0, axis=1)
        return newArray

    def _addNewDelta(self, newDelta: float):    
        self.deltas = np.append(self.deltas[1:], newDelta)

    def _addNewSmoothness(self, newDerivatives: np.ndarray):
        self.smoothness = (1 - self.alpha) * self.smoothness + self.alpha * newDerivatives

    def _updateSmoothnessMeasures(self):
        ## update all the smoothness measures
        self.smoothnessMeasures["overall"]["raw"] = np.sqrt(np.sum(self.smoothness[1:]**2))
        self.smoothnessMeasures["relative"]["raw"] = np.abs(self.smoothness[2] - self.smoothness[1]**2)
        self.smoothnessMeasures["secondOrder"]["raw"] = np.abs(self.smoothness[2])
        for measure in ["overall", "relative", "secondOrder"] :
            if (self.smoothnessMeasures[measure]["isCalibrated"]):
                self.smoothnessMeasures[measure]["calibratedSignal"] = sigmoid(
                    x=self.smoothnessMeasures[measure]["raw"], 
                    **self.smoothnessMeasures[measure]["calibration"])
        
        

    def addNewValue(self, newValue: float, newDelta: float=None, newTime: float=None):
        
        currentTime = time.time()*1000.0
        if newDelta is None and newTime is None:
            newDelta = currentTime - self.lastUpdateTime
            newTime = currentTime
        elif newDelta is None and newTime is not None:
            currentTime = newTime 
            newDelta = currentTime - self.lastUpdateTime
        elif newDelta is not None and newTime is None:
            currentTime = self.lastUpdateTime + newDelta
        else:
            raise Exception("Cant have both new Delta and new Time")



            
        self._addNewDelta(newDelta)
        newDerivatives = self._addNewDerivative(newValue, newDelta)
        self._addNewSmoothness(newDerivatives)
        self._updateSmoothnessMeasures()

        self.lastUpdateTime = newTime


    def getSmoothness(self, smoothnessMeasure="overall"):
        return {
            "raw": self.smoothnessMeasures[smoothnessMeasure]["raw"],
            "calibratedSignal": self.smoothnessMeasures[smoothnessMeasure]["calibratedSignal"]
        }
    
    def calibrateSignal(self, XMAX:float, XMIN:float, BETA:float=0.02, YMIN:float=0, YMAX:float=MIDI_MAX, smoothnessMeasure: str="overall"):
        self.smoothnessMeasures[smoothnessMeasure]["calibration"] = {
            "XMAX": XMAX, 
            "XMIN": XMIN, 
            "YMIN": YMIN,
            "YMAX": YMAX,
            "BETA": BETA}
        self.smoothnessMeasures[smoothnessMeasure]["isCalibrated"] = True 

    
    def decalibrateSignal(self, smoothnessMeasure: str="overall"):
        self.smoothnessMeasures[smoothnessMeasure]["isCalibrated"] = False 






class Smoothness():

    CHANNELS = {
        "CC16_1": "gyro x left",
        "CC16_2": "gyro x right",
        "CC17_1": "gyro y left",
        "CC17_2": "gyro y right",
        "CC18_1": "gyro z left",
        "CC18_2": "gyro z right",
        "CC19_1": "accel x left",
        "CC19_2": "accel x right",
        "CC20_1": "accel y left",
        "CC20_2": "accel y right",
        "CC21_1": "accel z left",
        "CC21_2": "accel z right",
        "CC22_1": "vel left",
        "CC22_2": "vel right",
    }

    SMOOTHNESS_TYPES = [
        "GYRO_LEFT",
        "GYRO_RIGHT",
        "VEL_LEFT",
        "VEL_RIGHT",
        "VEL_AVG",
        "ACCEL_LEFT",
        "ACCEL_RIGHT"
    ]
    
    NotAdmissibleType : Exception = Exception("Smoothness Type needs to be any of the admissible smoothness types: " + SMOOTHNESS_TYPES.__repr__())

    def __init__(
            self,
            smoothnessTypes: str or list[str],
            cacheLengths: None or int or list[int] = None, 
            derivativeDegrees: None or int or list[int] = None, 
            alphas: None or float or list[float] = None,
            customMidiMax: None or int = None):
        
        isNotAdmissibleSmoothnessType : bool = False
        if isinstance(smoothnessTypes, list):
            isNotAdmissibleSmoothnessType = any(t not in Smoothness.SMOOTHNESS_TYPES for t in smoothnessTypes)
        else:
            isNotAdmissibleSmoothnessType = smoothnessTypes not in Smoothness.SMOOTHNESS_TYPES
        
        if isNotAdmissibleSmoothnessType:
            raise Smoothness.NotAdmissibleType
        
        if customMidiMax is not None:
            self.midiMax = customMidiMax
        else:
            self.midiMax = MIDI_MAX

        if isinstance(smoothnessTypes, list):
            self.smoothnessTypes = smoothnessTypes
        else :
            self.smoothnessTypes = [smoothnessTypes]
        
        self.currentDefaultSmoothnessMeasure = "overall"

        self.data : dict[str, SmoothnessBase] = {}
        self.cacheLengths : dict[str, int] = {}
        self.derivativeDegrees : dict[str, int] = {}
        self.alphas : dict[str, float or list[float]] = {}


        if (cacheLengths is not None) and (derivativeDegrees is not None) and (alphas is not None): 
            if isinstance(cacheLengths,int):
                cacheLengths = [cacheLengths]*len(self.smoothnessTypes)
            if isinstance(derivativeDegrees, int):
                derivativeDegrees = [derivativeDegrees]*len(self.smoothnessTypes)
            if isinstance(alphas, float):
                alphas = [alphas] * len(self.smoothnessTypes)

            for tp, L, D, alpha in zip(self.smoothnessTypes, cacheLengths, derivativeDegrees, alphas):
                self.initSmoothnessType(tp, L, D, alpha)


    def initSmoothnessType(self, smoothnessType, cacheLength: int, derivativeDegree: int, alpha: float or list[float]):
        if smoothnessType not in Smoothness.SMOOTHNESS_TYPES:
            raise Smoothness.NotAdmissibleType
        if smoothnessType in self.data:
            raise Exception("Cannot add this type. It's already been added.")
        self.data[smoothnessType] = SmoothnessBase(cacheLength, derivativeDegree, alpha)
        self.cacheLengths[smoothnessType] = cacheLength
        self.derivativeDegrees[smoothnessType] = derivativeDegree
        self.alphas[smoothnessType] = alpha


    def removeSmoothnessType(self, smoothnessType):
        if smoothnessType in self.data:
            del self.data[smoothnessType] 
            del self.cacheLengths[smoothnessType] 
            del self.derivativeDegrees[smoothnessType] 
            del self.alphas[smoothnessType]     


    def addNewValues(self, 
                     channelData: dict[int, float],
                     newTime: float = None, 
                     newDelta: float = None, 
                     returnSmoothness: bool = False):
        for smoothnessType in self.data.keys():
            newValue = self.conversion(smoothnessType, channelData)
            self.data[smoothnessType].addNewValue(newValue, newTime=newTime, newDelta=newDelta)
        if returnSmoothness:
            return self.getSmoothnessMeasure()
        

    def getSmoothnessMeasure(self, smoothnessMeasure=None):
        smoothnessMeasure = (self.currentDefaultSmoothnessMeasure if smoothnessMeasure is None else smoothnessMeasure)
        return  { 
            smoothnessType: (
                {
                    "raw": smoothnessObject.smoothnessMeasures[smoothnessMeasure]["raw"],
                    "calibratedSignal": smoothnessObject.smoothnessMeasures[smoothnessMeasure]["calibratedSignal"]
                }
                if smoothnessObject.smoothnessMeasures[smoothnessMeasure]["isCalibrated"]
                else {"raw": smoothnessObject.smoothnessMeasures[smoothnessMeasure]["raw"]})
            for smoothnessType, smoothnessObject in self.data.items()}
        

    def calibrateSmoothness(self, smoothnessTypes: str or list[str], minsAndMaxs: list[int] or list[list[int]], smoothnessMeasure: str="overall"):
        if isinstance(smoothnessTypes, str):
            self.data[smoothnessTypes].calibrateSignal(XMIN=minsAndMaxs[0], XMAX=minsAndMaxs[1], smoothnessMeasure=smoothnessMeasure)
        else:
            if len(smoothnessTypes)!=len(minsAndMaxs):
                raise Exception("arrays should be of equal lengths")
            for sT, minMAx in zip(smoothnessTypes, minsAndMaxs):
                self.data[sT].calibrateSignal(XMIN=minMAx[0], XMAX=minMAx[1], smoothnessMeasure=smoothnessMeasure)


    def changeCurrentDefaultSmoothnessMeasure(self, newSmoothnessMeasure="overall"):
        self.currentDefaultSmoothnessMeasure = newSmoothnessMeasure


    def conversion(self, smoothnessType: str, data: dict[int, float]):
        if (smoothnessType=="GYRO_LEFT"):

            theta = data["CC16_1"] * 2 * np.pi / self.midiMax
            phi = data["CC17_1"] * 2 * np.pi / self.midiMax
            psi = data["CC18_1"] * 2 * np.pi / self.midiMax
            # y = np.sin(theta)
            # z = np.cos(theta)
            # x = np.cos(phi)
            # z = np.sin(phi)
            # x = np.sin(psi)
            # y = np.cos(psi)
            x = np.cos(phi) + np.sin(psi)
            y = np.sin(theta) + np.cos(psi)
            z = np.cos(theta) + np.sin(phi)
            return np.sqrt(x**2 + y**2 + z**2)
        elif (smoothnessType=="GYRO_RIGHT"):
            theta = data["CC16_2"] * 2 * np.pi / self.midiMax
            phi = data["CC17_2"] * 2 * np.pi / self.midiMax
            psi = data["CC18_2"] * 2 * np.pi / self.midiMax
            x = np.cos(phi) + np.sin(psi)
            y = np.sin(theta) + np.cos(psi)
            z = np.cos(theta) + np.sin(phi)
            return np.sqrt(x**2 + y**2 + z**2)
        elif (smoothnessType=="VEL_LEFT"):
            return data["CC22_1"] / self.midiMax
        elif (smoothnessType=="VEL_RIGHT"):
            return data["CC22_2"] / self.midiMax
        elif (smoothnessType=="VEL_AVG"):
            return (data["CC22_1"] + data["CC22_2"] ) / (2 * self.midiMax)
        if (smoothnessType=="ACCEL_LEFT"):
            ax = data["CC19_1"]  / self.midiMax
            ay = data["CC20_1"]  / self.midiMax
            az = data["CC21_1"]  / self.midiMax
            return np.sqrt(ax**2 + ay**2 + az**2)
        elif (smoothnessType=="ACCEL_RIGHT"):
            ax = data["CC19_2"] / self.midiMax
            ay = data["CC20_2"] / self.midiMax
            az = data["CC21_2"] / self.midiMax
            return np.sqrt(ax**2 + ay**2 + az**2)
        
        


