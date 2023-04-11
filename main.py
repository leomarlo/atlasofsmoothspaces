import numpy as np


class SmoothnessBase():

    
    def __init__(self, 
        cacheLength: int,
        derivativeDegree: int,
        alpha: float or list[float]):

        self.cacheLength = cacheLength
        self.derivativeDegree = derivativeDegree
        self.smoothness = np.zeros(self.derivativeDegree + 1)
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

    def addNewValue(self, newValue: float, newDelta: float):
        self._addNewDelta(newDelta)
        newDerivatives = self._addNewDerivative(newValue, newDelta)
        self._addNewSmoothness(newDerivatives)



class Smoothness():

    MIDI_MAX : int = 137

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
            alphas: None or float or list[float] = None):
        
        isNotAdmissibleSmoothnessType : bool = False
        if isinstance(smoothnessTypes, list):
            isNotAdmissibleSmoothnessType = any(t not in Smoothness.SMOOTHNESS_TYPES for t in smoothnessTypes)
        else:
            isNotAdmissibleSmoothnessType = smoothnessTypes not in Smoothness.SMOOTHNESS_TYPES
        
        if isNotAdmissibleSmoothnessType:
            raise Smoothness.NotAdmissibleType
        

        if isinstance(smoothnessTypes, list):
            self.smoothnessTypes = smoothnessTypes
        else :
            self.smoothnessTypes = [smoothnessTypes]
        
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


    def addNewValues(self, channelData: dict[int, float]):
        pass



    def conversion(self, smoothnessType: str, data: dict[int, float]):
        if (smoothnessType=="GYRO_LEFT"):
            return data["CC16_1"] * 2 * np.pi / Smoothness.MIDI_MAX
        
        "GYRO_LEFT",
        "GYRO_RIGHT",
        "VEL_LEFT",
        "VEL_RIGHT",
        "VEL_AVG",
        "ACCEL_LEFT",
        "ACCEL_RIGHT"
    
        


