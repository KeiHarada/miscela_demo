class Color:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[36m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'

class Sensor:

    # constructor
    def __init__(self):
        self._attribute = str()
        self._id = str()
        self._lat = float()
        self._lon = float()
        self._time = list()
        self._data = list()
        self._neighbor = set()
        self._tp = set()
        self._tn = set()

    # setter
    def setId(self, id):
        self._id = id
    def setAttribute(self, attribute):
        self._attribute = attribute
    def setLocation(self, location):
        lat, lon = location
        self._lat = lat
        self._lon = lon
    def setData(self, data):
        self._data = data
    def setTime(self, time):
        self._time = time
    def addNeighbor(self, neighbor):
        self._neighbor.add(neighbor)
    def addTp(self, t):
        self._tp.add(t)
    def addTn(self, t):
        self._tn.add(t)

    # accessor
    def getId(self):
        return self._id
    def getAttribute(self):
        return self._attribute
    def getLocation(self):
        return [self._lat, self._lon]
    def getData(self):
        return self._data
    def getTime(self):
        return self._time
    def getNeighbor(self):
        return self._neighbor
    def getTp(self):
        return self._tp
    def getTn(self):
        return self._tn

class Cluster:

    # constructor
    def __init__(self):
        self._member = list()
        self._attribute = set()

    # setter
    def setMember(self, mem):
        self._member = mem
    def setAttribute(self, att):
        self._attribute = att

    # accessor
    def getMember(self):
        return self._member
    def getAttribute(self):
        return self._attribute

class CAP:

    # constructor
    def __init__(self):
        self._id = int()
        self._attribute = set()
        self._pattern = dict()
        self._member = list()
        self._coevolution = set()
        self._p1 = set()
        self._p2 = set()

    # setter
    def setId(self, id):
        self._id = id
    def addAttribute(self, att):
        self._attribute.add(att)
        self._pattern[att] = 0
    def addMember(self, sensor):
        self._member.append(sensor)
    def setCoevolution(self, timestamp):
        self._coevolution = timestamp
    def setPattern(self, att, p):
        self._pattern[att] = p
    def setP1(self, t):
        self._p1 = t
    def setP2(self, t):
        self._p2 = t

    # accessor
    def getId(self):
        return self._id
    def getAttribute(self):
        return self._attribute
    def getCoevolution(self):
        return self._coevolution
    def getMember(self):
        return self._member
    def getPattern(self):
        return self._pattern
    def getP1(self):
        return self._p1
    def getP2(self):
        return self._p2
