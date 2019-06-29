import pandas as pd
import numpy as np
import copy
import pickle
from pyclustering.cluster.dbscan import dbscan

from src.myclass import Color
from src.myclass import Sensor
from src.myclass import Cluster
from src.myclass import CAP
from src.myutility import deg2km
from src.myutility import dist

def loadData(attribute, dataset):

    data = pd.read_csv("db/"+dataset+"/data.csv", dtype=object)
    data = data[data["attribute"] == attribute]
    location = pd.read_csv("db/"+dataset+"/location.csv", dtype=object)
    location = location[location["attribute"] == attribute]
    ids = list(location["id"])
    timestamps = list(data["time"])

    s = list()
    for i in ids:
        location_i = location[location["id"] == str(i)]
        location_i = (float(location_i["lat"]), float(location_i["lon"]))
        data_i = data[data["id"] == str(i)]
        data_i = list(data_i["data"])
        s_i = Sensor()
        s_i.setId(str(i))
        s_i.setAttribute(str(attribute))
        s_i.setTime(timestamps)
        s_i.setLocation(location_i)
        s_i.setData(data_i)
        s.append(s_i)
        del s_i

    return s

def dataSegmenting(S):

    '''
    algorithm
    rpt.Dynp(model='l2', custom_cost=None, min_size=2, jump=5, params=None)
    rpt.Pelt(model='l2', custom_cost=None, min_size=2, jump=5, params=None)
    rpt.Binseg(model='l2', custom_cost=None, min_size=2, jump=5, params=None)
    rpt.BottomUp(model='l2', custom_cost=None, min_size=2, jump=5, params=None)
    rpt.Window(width=100, model='l2', custom_cost=None, min_size=2, jump=5, params=None)
    '''

    for s_i in S:
        data = pd.Series(s_i.getData())
        data = data.fillna(method="ffill")
        data = data.fillna(method="bfill")
        data = data.astype("float64")
        s_i.setData_filled(list(data))

def estimateThreshold(S, M, evoRate):

    thresholds = dict()

    # each attribute
    offset = 0
    for attribute in M.keys():
        distribution = list()

        # each sensor
        for s_i in S[offset: offset+M[attribute]]:
            data = s_i.getData_filled()
            prev = 0.0

            # each value
            for value in data:
                distribution.append(abs(value-prev))
                prev = value

        distribution.sort(reverse=True)
        threshold = distribution[int(evoRate * len(distribution))]
        thresholds[attribute] = threshold
        offset += M[attribute]
        del distribution

    return thresholds

def extractEvolving(S, thresholds):

    for s in S:
        prev = 0.0
        data = s.getData_filled()
        for i in range(len(data)):
            delta = data[i] - prev
            if delta > thresholds[s.getAttribute()]:
                s.addTp(i)
            if delta < (-1)*thresholds[s.getAttribute()]:
                s.addTn(i)
            prev = data[i]

def clustering(S, distance):

    '''
    DBSCAN
    '''

    locations = list(map(lambda s_i: deg2km(s_i.getLocation()[0], s_i.getLocation()[1]), S))
    inst = dbscan(data=locations, eps=distance, neighbors=2, ccore=False) # True is for C++, False is for Python.
    inst.process()
    clusters = inst.get_clusters()

    '''
    set the results into Cluster class
    '''

    C = list()
    for cluster in clusters:
        c = Cluster()
        cluster.sort()
        c.setMember(cluster)
        attributes = set()
        for i in cluster:
            attributes.add(S[i].getAttribute())
            for j in cluster[cluster.index(i)+1:]:
                    if dist(S[i].getLocation(), S[j].getLocation()) <= distance:
                        S[i].addNeighbor(j)
                        S[j].addNeighbor(i)

        c.setAttribute(attributes)
        C.append(c)

    return C

def capSearch(S, C, K, psi):

    CAPs = list()
    for c in C:
        CAPs += search(S, c, K, psi, list(), list())

    for i in range(len(CAPs)):
        CAPs[i].setId(i)
        CAPs[i].setCoevolution()

    return CAPs

def search(S, c, K, psi, X, CAP_X):

    CAPs = list()

    if len(X) >= 2:
        CAPs += CAP_X

    F_X = follower(S, c, X)

    for y in F_X:
        Y = X.copy()
        Y.append(y)
        Y.sort()

        if parent(S, Y, K) == X:
            CAP_Y = getCAP(S, y, psi, CAP_X)
            if len(CAP_Y) != 0:
                CAPs += search(S, c, K, psi, Y, CAP_Y)

    return CAPs

def follower(S, c, X):

    # root
    if len(X) == 0:
        return c.getMember()

    # followers
    else:
        F_X = set()
        for x in X:
            F_X |= S[x].getNeighbor()
        F_X -= set(X)
        return sorted(list(F_X))

def parent(S, Y, K):

    # size(Y) = 1
    if len(Y) == 1:
        return list()

    # size(Y) == 2
    if len(Y) == 2:
        if S[Y[0]].getAttribute() == S[Y[1]].getAttribute():
            return list()
        else:
            return [Y[1], ]

    # size(Y) >= 3
    # Y contains more/less than or equal to 2/K attributes
    attCounter = set()
    for y in Y:
        attCounter.add(S[y].getAttribute())
    if len(attCounter) > K:
        return list()

    for y in Y:
        Z = Y.copy()
        Z.remove(y)
        L_Z = np.array([[0]*len(Z)]*len(Z))
        for i in range(0, len(Z)):
            for j in range(i+1, len(Z)):
                if Z[j] in S[Z[i]].getNeighbor():
                    L_Z[i][j] = -1
                    L_Z[j][i] = -1
            L_Z[i][i] = np.count_nonzero(L_Z[i])

        # rank(L(Z)) = |Z|-1 => Z is connected
        if np.linalg.matrix_rank(L_Z) == len(Z)-1:

            # Z contains more/less than or equal to 2/K attributes
            attCounter = set()
            for z in Z:
                attCounter.add(S[z].getAttribute())
            if len(attCounter) >= 2 and len(attCounter) <= K:
                return Z

    return list()

def getCAP(S, y, psi, C_X):

    C_Y = list()

    # init
    if len(C_X) == 0:
        if len(S[y].getTp()) + len(S[y].getTn()) >= psi:
            cap = CAP()
            cap.addMember(y)
            cap.addAttribute(S[y].getAttribute())
            cap.setPattern(S[y].getAttribute(), 1)
            cap.setP1(S[y].getTp())
            cap.setP2(S[y].getTn())
            C_Y.append(cap)

        return C_Y

    # following
    else:

        for cap_x in C_X:
            cap = copy.deepcopy(cap_x)
            p1 = set()
            p2 = set()

            # y_a isn't a new attribute
            if S[y].getAttribute() in cap.getAttribute():

                # calculate intersection (1:increase, -1:decrease)
                if cap.getPattern()[S[y].getAttribute()] == 1:
                    p1 = cap.getP1() & S[y].getTp()
                    p2 = cap.getP2() & S[y].getTn()
                if cap.getPattern()[S[y].getAttribute()] == -1:
                    p1 = cap.getP1() & S[y].getTn()
                    p2 = cap.getP2() & S[y].getTp()
                if cap.getPattern()[S[y].getAttribute()] == 0:
                    print("cap error")
                    quit()

                # set cap
                if len(p1)+len(p2) >= psi:
                    cap.addMember(y)
                    cap.setP1(p1)
                    cap.setP2(p2)
                    C_Y.append(cap)

            # y_a is a new attribute
            else:

                cap_new = copy.deepcopy(cap)
                p1 = cap_new.getP1() & S[y].getTp()
                p2 = cap_new.getP2() & S[y].getTn()
                if len(p1) + len(p2) >= psi:
                    cap_new.addAttribute(S[y].getAttribute())
                    cap_new.addMember(y)
                    cap_new.setPattern(S[y].getAttribute(), 1)
                    cap_new.setP1(p1)
                    cap_new.setP2(p2)
                    C_Y.append(cap_new)

                del cap_new

                cap_new = copy.deepcopy(cap)
                p1 = cap_new.getP1() & S[y].getTn()
                p2 = cap_new.getP2() & S[y].getTp()
                if len(p1) + len(p2) >= psi:
                    cap_new.addAttribute(S[y].getAttribute())
                    cap_new.addMember(y)
                    cap_new.setPattern(S[y].getAttribute(), -1)
                    cap_new.setP1(p1)
                    cap_new.setP2(p2)
                    C_Y.append(cap_new)

        return C_Y

def outputCAP(dataset, S, CAPs):

    for cap in CAPs:

        cap_id = cap.getId()

        with open("result/" + dataset + "/" + str(cap_id).zfill(5) + "_pattern.csv", "w") as of_pattern:
            with open("result/"+dataset+"/"+str(cap_id).zfill(5)+"_location.csv", "w") as of_location:
                with open("result/" + dataset + "/" + str(cap_id).zfill(5) + "_data.csv", "w") as of_data:
                    with open("result/" + dataset + "/" + str(cap_id).zfill(5) + "_data_filled.csv", "w") as of_data_filled:

                        of_pattern.write("id,attribute,pattern\n")
                        of_location.write("id,attribute,lat,lon\n")
                        data = pd.DataFrame(S[0].getTime(), columns=["time"])
                        data_filled = pd.DataFrame(S[0].getTime(), columns=["time"])

                        for i in cap.getMember():

                            sid = S[i].getId()
                            attribute = S[i].getAttribute()

                            # pattern
                            pattern = cap.getPattern()[attribute]
                            of_pattern.write(sid+","+attribute+","+str(pattern)+"\n")

                            # location
                            lat, lon = S[i].getLocation()
                            of_location.write(sid+","+attribute+","+str(lat)+","+str(lon)+"\n")

                            # data
                            data[sid] = pd.Series(S[i].getData())
                            data_filled[sid] = pd.Series(S[i].getData_filled())

                        data.to_csv(of_data, index=False)
                        data_filled.to_csv(of_data_filled, index=False)

def miscela(args):

    print("*----------------------------------------------------------*")
    print("* MISCELA is getting start ...")

    # load data on memory
    print("\t|- phase0: loading data ... ", end="")
    S = list()
    M = dict()
    for attribute in list(open("db/"+str(args.dataset)+"/attribute.csv", "r").readlines()):
        attribute = attribute.strip()
        S_a = loadData(attribute, str(args.dataset))
        S += S_a
        M[attribute] = len(S_a)
        del S_a
    print(Color.GREEN + "OK" + Color.END)

    # data segmenting
    print("\t|- phase1: pre-processing ... ", end="")
    dataSegmenting(S)
    print(Color.GREEN + "OK" + Color.END)

    # extract evolving timestamps
    print("\t|- phase2: extracting evolving timestamps ... ", end="")
    thresholds = estimateThreshold(S, M, args.evoRate)
    extractEvolving(S, thresholds)
    print(Color.GREEN + "OK" + Color.END)

    # clustering
    print("\t|- phase3: clustering ... ", end="")
    C = clustering(S, args.distance)
    print(Color.GREEN + "OK" + Color.END)

    # CAP search
    print("\t|- phase4: cap search ... ", end="")
    CAPs = capSearch(S, C, args.maxAtt, args.minSup)
    print(Color.GREEN + "OK" + Color.END)

    # output
    print(len(CAPs))
    # outputCAP(args.dataset, S, CAPs)
    with open("pickle/"+args.dataset+"/sensor.pickle", "wb") as pl:
        pickle.dump(S, pl)
    with open("pickle/"+args.dataset+"/attribute.pickle", "wb") as pl:
        pickle.dump(M, pl)
    with open("pickle/"+args.dataset+"/cluster.pickle", "wb") as pl:
        pickle.dump(C, pl)
    with open("pickle/"+args.dataset+"/cap.pickle", "wb") as pl:
        pickle.dump(CAPs, pl)
    with open("pickle/"+args.dataset+"/threshold.pickle", "wb") as pl:
        pickle.dump(thresholds, pl)


def mocServer(args):

    cap_id = 0
    example = {"00202": "temperature",
               "00199": "temperature",
               "00197": "temperature",
               "00064": "temperature",
               "00203": "temperature",
               "00193": "temperature",
               "10029": "light",
               "10126": "light",
               "10171": "light",
               "10129": "light",
               "10099": "light"}

    with open("result/"+args.dataset+"/"+str(cap_id).zfill(5)+"_pattern.csv", "w") as outfile:
        outfile.write("id,attribute,pattern\n")
        for sensor_id, attribute in example.items():
            outfile.write(sensor_id+","+attribute+",1\n")

    with open("result/"+args.dataset+"/"+str(cap_id).zfill(5)+"_location.csv", "w") as outfile:
        outfile.write("id,attribute,lat,lon\n")
        for sensor_id, attribute in example.items():
            with open("db/" + args.dataset + "/location.csv", "r") as infile:
                for line in infile.readlines()[1:]:
                    _sensor_id, _atttribute, _lat, _lon = line.strip().split(",")
                    if _sensor_id == sensor_id:
                        outfile.write(line)

    with open("result/"+args.dataset+"/"+str(cap_id).zfill(5)+"_data.csv", "w") as outfile1:
        with open("result/" + args.dataset + "/" + str(cap_id).zfill(5) + "_data_filled.csv", "w") as outfile2:
            outfile1.write("id,attribute,lat,lon\n")
            outfile2.write("id,attribute,lat,lon\n")

            ids = dict()
            with open("db/"+args.dataset+"/data.csv", "r") as infile:
                outfile1.write("time")
                outfile2.write("time")
                for sensor_id in example.keys():
                    ids[sensor_id] = []
                    outfile1.write("," + sensor_id)
                    outfile2.write("," + sensor_id)
                outfile1.write("\n")
                outfile2.write("\n")

            for sensor_id in ids.keys():
                timestamp = []
                with open("db/"+args.dataset+"/data.csv", "r") as infile:
                    for line in infile.readlines()[1:]:
                        _sensor_id, _attribute, _time, _data = line.strip().split(",")
                        if _sensor_id == sensor_id:
                            ids[_sensor_id].append(_data)
                            timestamp.append(_time)

            for i in range(len(timestamp)):
                outfile1.write(timestamp[i])
                for sensor_id in ids.keys():
                    outfile1.write("," + ids[sensor_id][i])
                outfile1.write("\n")

            for sensor_id in ids:
                ids[sensor_id] = [np.nan if i == "null" else i for i in ids[sensor_id]]
                data = pd.Series(ids[sensor_id])
                data = data.fillna(method="ffill")
                data = data.fillna(method="bfill")
                ids[sensor_id] = list(data)

            for i in range(len(timestamp)):
                outfile2.write(timestamp[i])
                for sensor_id in ids.keys():
                    outfile2.write("," + ids[sensor_id][i])
                outfile2.write("\n")