
import openmesh as om
import os
import numpy as np
import csv
import math as Math
import time
#d3v imports
from iohandlers import IOHandler
from signals import Signals
from geometry import Geometry

# import matplotlib.pyplot as plt

class HullFormMeshQuality:
    def __init__(self):
        self._numWL = 20
        self._numPnWLhalf = 20
        self._distPolyOrder=3

    @property
    def numPointsWLhalf(self):
        return self._numPnWLhalf

    def _getDistribution(self, maxv, minv, n, pot):
        x = [0.0] * n
        for i in range(n):
            fi = float(i)
            fn1 = float(n - 1)
            x[i] = fi ** pot / fn1 ** pot * (maxv - minv) + minv
        x.reverse()
        return x

    def genWLPositions(self,hWL_top,  hWL_bottom ):
        wlPos = self._getDistribution(hWL_top, hWL_bottom, self._numWL, self._distPolyOrder)
        return wlPos
    def genWLPositionsUsingObligatory(self,obligatoryLines:list ):
        testLines = self._getDistribution(obligatoryLines[0], obligatoryLines[-1], self._numWL, self._distPolyOrder)

        nol=len(obligatoryLines)
        wlPos=[]
        i1TL=0
        for iol in range(1,nol):
            hmax=obligatoryLines[iol-1]
            hmin = obligatoryLines[iol]
            numWL=0
            for iTL in range(i1TL,self._numWL):
                if testLines[iTL] < hmax:
                    if testLines[iTL] > hmin:
                        numWL =numWL + 1
                    else:
                        i1TL=iTL
                        break
            wlPosi = self._getDistribution(hmax, hmin, numWL+2, 1)
            for wl in wlPosi:
                if len(wlPos)==0:
                    wlPos.append(wl)
                elif wl < wlPos[-1]:
                    wlPos.append(wl)
        return wlPos


class HullForm(Geometry):
    def __init__(self, fileName):
        super().__init__()
        self.filename = fileName
        self.hfmq = HullFormMeshQuality()
        self.shipdata,self.pdecks,self.pbulkheads = self.readShipData()

        self.h = []  # positive y waterlines
        self.wlinesNeg = []  # negative y waerlines
        self.wlKeel = []  # keel waterline (one waterline)
        self.generateMesh()

    def generateMesh(self):

        hmax=self.pdecks[0]
        #hmax=9.4
        #wlPos=self.hfmq.genWLPositions(hmax, 0)
        transomTip = self.shipdata["draft_val"] * self.shipdata["td_val"]
        obligatoryWL= []
        for dh in self.pdecks:
            obligatoryWL.append(dh)
        obligatoryWL.append(transomTip)
        obligatoryWL.sort(reverse=True)

        wlPos = self.hfmq.genWLPositionsUsingObligatory(obligatoryWL)
        #results = self.hullGen(self.shipdata, wlPos, self.hfmq.numPointsWLhalf)
        lines = self.hullGen(self.shipdata, wlPos, self.hfmq.numPointsWLhalf)
        self.wlinesPos = lines[0]  # positive y waterlines
        self.wlinesNeg = lines[1]  # negative y waerlines
        self.wlKeel = lines[2]  # keel waterline (one waterline)
        self.mesh = self.genHullFormMeshPP(lines)
        pass


    def getResultsOld(self,h,seaDensity):
        tsStart = time.perf_counter()
        results = []
        fvs = self.mesh.fv_indices().tolist()
        points = self.mesh.points().tolist()
        xmf = 50
        # h=9
        bcwl = self.getBasicDataUsingTrianglesProjectedToWaterline(h,xmf,fvs,points)
        h = bcwl[0]
        volume = bcwl[1]
        area = bcwl[2]
        Xwl = bcwl[3]
        KBz = bcwl[4]
        KBx = bcwl[5]
        Ib = bcwl[6]
        Il = bcwl[7]
        Lwl = bcwl[8]
        Bwl = bcwl[9]
        mfarea = bcwl[10]


        hsdata = self.getHydrostaticData(seaDensity,h,volume,area,Ib, Il,KBz,Lwl,Bwl,mfarea)
        dtAll = time.perf_counter() - tsStart
        print("Hydrostatic results calc time:", dtAll)
        results = bcwl+hsdata
        print(results)
        return results

    def getResults(self,h,seaDensity):
        tsStart = time.perf_counter()
        results = []
        fvs = self.mesh.fv_indices().tolist()
        points = self.mesh.points().tolist()
        xmf = 50
        # h=9
        bcwl = self.getBasicDataUsingTrianglesProjectedToWaterline(h,xmf,fvs,points)
        h = bcwl[0]
        volume = bcwl[1]
        area = bcwl[2]
        Xwl = bcwl[3]
        KBz = bcwl[4]
        KBx = bcwl[5]
        Ib = bcwl[6]
        Il = bcwl[7]
        Lwl,Bwl = self.getLwlBwl(h,fvs,points)
        mfarea = self.getMainFrameArea(xmf, h, fvs, points)
        hsdata = self.getHydrostaticData(seaDensity,h,volume,area,Ib,Il,KBz,Lwl,2*Bwl,mfarea)
        dtAll = time.perf_counter() - tsStart
        print("Hydrostatic results calc time:", dtAll)
        results = bcwl+hsdata
        print(results)
        return results

    def getMainFrameArea(self,x,h,fvs,points):
        mfpoints = self.getSortedPointsOnAxisAlignedPlane(x, fvs, points, 0)
        area=0
        a = len(mfpoints)
        for i in range(0,a-2,1):
                h1 = mfpoints[i][2]
                h2 = mfpoints[i+2][2]
                b1 = mfpoints[i][1]
                b2 = mfpoints[i+2][1]
                if h1<=h and h2<=h:
                    area = area + 1/2*abs(h2-h1)*(abs(b1)+abs(b2))
                if h2>h and h1<h:
                    point = self.getIntersectionPoint(mfpoints[i+2],mfpoints[i],h,2)
                    H2 = point[2]
                    area = area + 1/2*abs(H2-h1)*(abs(b1)+abs(b2))

        return area

    def getSortedPointsOnAxisAlignedPlane(self, x, fvs, points, os):
        mfpoints=[]
        lpr = []
        lpl = []
        for fv in fvs:
            lpr.clear()
            lpl.clear()
            for iv in fv:
                p = points[iv]
                if p[os] < x:
                    lpl.append(iv)
                elif p[os] > x:
                    lpr.append(iv)
                else:
                    mfpoints.append(p)

            if len(lpl)>0 and len(lpr) > 0:
                if len(lpl) < len(lpr):
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]],points[lpr[0]],x,os))
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]], points[lpr[1]], x, os))
                elif len(lpl) > len(lpr):
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]],points[lpr[0]],x,os))
                    mfpoints.append(self.getIntersectionPoint(points[lpl[1]], points[lpr[0]], x, os))
                else:
                    mfpoints.append(self.getIntersectionPoint(points[lpl[0]],points[lpr[0]],x,os))
                pass

        #mfpoints=[[0,1,1],[0,2,21],[1,1,2],[10,0,0]]


        import itertools
        mfpoints.sort()
        mftemp = list(mfpoints for mfpoints,_ in itertools.groupby(mfpoints))
        mfpoints = mftemp

        mfpoints = sorted(mfpoints, key=lambda p: p[2])
        return mfpoints

    def getHydrostaticData(self,seaDensity,h,volume,area,Ib, Il, KBz,Lwl,Bwl,mfarea):

        MoB = Ib / volume
        KMo = MoB + KBz
        MlB = Il / volume
        KMl = MlB + KBz
        JZ = 0.01 * area * seaDensity
        M1 = Il / Lwl
        delta = volume / seaDensity

        Cwl = area / (Lwl * Bwl)
        CB = volume / (Lwl * Bwl * h)

        CP = volume / (mfarea * Lwl)
        CX = mfarea / (Bwl * h)
        return KMo, KMl, JZ, M1, delta, Cwl, CB, CP, CX

    def getHydrostaticDataOld(self,seaDensity,h,volume,area,Ib, Il, KBz,Lwl,Bwl,mfarea):

        MoB = Ib / volume
        KMo = MoB + KBz
        MlB = Il / volume
        KMl = MlB + KBz
        JZ = 0.01 * area * seaDensity

        Cwl = area / (Lwl * Bwl)
        CB = volume / (Lwl * Bwl * h)

        CP = volume / (mfarea * Lwl)
        CX = mfarea / (Bwl * h)
        # results = [MoB,KMo,MlB,KMl,JZ,Cwl,CB,CP,CX]

        return MoB,KMo,MlB,KMl,JZ,Cwl,CB,CP,CX

    def getLwlBwl(self, h, fvs, points):
        wlpoints = self.getSortedPointsOnAxisAlignedPlane(h, fvs, points, 2)
        wlpoints = sorted(wlpoints, key=lambda p: p[1])
        Bwl = wlpoints[-1][1]-wlpoints[0][1]

        wlpoints = sorted(wlpoints, key=lambda p: p[0])

        Lwl = wlpoints[-1][0] - wlpoints[0][0]
        return Lwl, Bwl

    def getBasicDataUsingTrianglesProjectedToWaterline(self, h, x, fvs, points):
        mesh = self.mesh
        lpowl = []
        lpbwl = []
        Ib = 0
        KBz = 0
        KBx = 0
        vol = 0
        Awl = 0
        Xwl = 0
        Swet=0
        p = []
        r = []
        cgTriaMid = []
        a = self.getXwl(h)
        for fh in fvs:  # facet handle
            p.clear()
            r.clear()
            cgTriaMid.clear()
            lpowl.clear()
            lpbwl.clear()
            i = 0
            for vh in fh:  # vertex handle
                p.append(points[vh])
                if p[i][2] > h:
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i = i + 1
            if len(lpowl) < 1:
                # A
                # Ax = points[fvs[fh][0]][0]
                Ax = p[0][0]
                Ay = p[0][1]
                Az = p[0][2]
                # B
                Bx = p[1][0]
                By = p[1][1]
                Bz = p[1][2]
                # C
                Cx = p[2][0]
                Cy = p[2][1]
                Cz = p[2][2]
                areaXYPlane = self.calcArea2DTria(Ax, Ay, Bx, By, Cx, Cy)
                area3D = self.calcAreaTria3D(p[0],p[1],p[2])
                Swet+=area3D

                # Volume
                hA = h - Az
                hB = h - Bz
                hC = h - Cz
                vol = vol + 1 / 3 * abs(areaXYPlane) * (hA + hB + hC)

                # Ib Il
                r.append(self.TezisteTrokuta(Ax, Ay, h, Bx, By, h, Cx, Cy, h))
                Ib = Ib + r[0][1] ** 2 * areaXYPlane

                # Kbz, KBx
                hsr = (Az + Bz + Cz) / 3
                cgTriaMid.append(self.TezisteTrokuta(Ax, Ay, (h - hsr), Bx, By, (h - hsr), Cx, Cy, (h - hsr)))
                KBz = KBz + abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))) * (h - hsr) * (
                        hsr + (h - hsr) / 2)
                KBx = KBx + abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))) * (h - hsr) * (
                    cgTriaMid[0][0])

                # Awl
                area = self.calcArea2DTria(Ax, Ay, Bx, By, Cx, Cy)

                # Xwl
                Xwl = Xwl + (r[0][0] * abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))))

                pass
            elif len(lpowl) == 1:
                # 2 trokuta Ib, Il i Xwl
                lip = self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]], h, 2)
                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[1]][0],
                                             p[lpbwl[1]][1], h))
                Ib = Ib + r[0][1] ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0],
                                                             p[lpbwl[1]][1])

                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0],
                                                          p[lpbwl[1]][1])

                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h, p[lpbwl[0]][0],
                                             p[lpbwl[0]][1], h))
                Ib = Ib + r[1][1] ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1],
                                                             p[lpbwl[0]][0], p[lpbwl[0]][1])

                Xwl = Xwl + r[1][0] * self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1],
                                                          p[lpbwl[0]][0], p[lpbwl[0]][1])

                # Kbz, KBx
                hsr = (lip[0][2] + lip[1][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[1]][0],
                                                     p[lpbwl[1]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0],
                                                p[lpbwl[1]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0],
                                                p[lpbwl[1]][1]) * (h - hsr) * (cgTriaMid[0][0])

                hsr = (lip[0][2] + p[lpbwl[1]][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(
                    self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h, p[lpbwl[0]][0],
                                        p[lpbwl[0]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0],
                                                p[lpbwl[0]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0],
                                                p[lpbwl[0]][1]) * (h - hsr) * (cgTriaMid[0][0])

                # Volume
                area1 = self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[0]][0], p[lpbwl[0]][1], p[lpbwl[1]][0],
                                            p[lpbwl[1]][1])
                area2 = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                areaXYPlane = area1 + area2

                vol = vol + 1 / 3 * (h - p[lpbwl[0]][2] + h - p[lpbwl[1]][2]) * area1
                vol = vol + 1 / 3 * (h - p[lpbwl[1]][2]) * area2

                area3D = self.calcAreaTria3D(lip[0], lpbwl[0], lpbwl[1]) + self.calcAreaTria3D(lip[0], lip[1], lpbwl[1])
                Swet += area3D
                # Awl
                area = area1 + area2

                pass
            elif len(lpowl) == 2:
                # 1 trokut Ib, Il
                lip = self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]], h, 2)
                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[0]][0],
                                             p[lpbwl[0]][1], h))
                Ib = Ib + r[0][1] ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],
                                                             p[lpbwl[0]][1])

                # KBz, KBx
                hsr = (lip[0][2] + lip[1][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[0]][0],
                                                     p[lpbwl[0]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],
                                                p[lpbwl[0]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],
                                                p[lpbwl[0]][1]) * (h - hsr) * (cgTriaMid[0][0])

                # Volume
                areaXYPlane = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],
                                                  p[lpbwl[0]][1])

                vol = vol + 1 / 3 * (h - p[lpbwl[0]][2]) * areaXYPlane

                area3D = self.calcAreaTria3D(lip[0], lip[1], lpbwl[0])
                Swet += area3D

                # Awl
                area = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                # Xwl
                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],
                                                          p[lpbwl[0]][1])

                pass
            else:
                area = 0
            Awl = Awl + area

        Xwl = Xwl / Awl
        Il = float(self.getIl(h, Xwl))
        KBz = KBz / vol
        KBx = KBx / vol
        #Lwl, Bwl = self.getLwlBwl(h, fvs, points)
        #mfarea = self.getMainFrameArea(x, h, fvs, points)

        # dodati Swet u izlaz
        return h, 2 * vol, 2 * Awl, Xwl, KBz, KBx, 2 * Ib, 2 * Il

    def getBasicDataUsingTrianglesProjectedToWaterlineOld(self,h,x,fvs,points):
        mesh = self.mesh
        lpowl = []
        lpbwl = []
        Ib = 0
        KBz = 0
        KBx = 0
        vol = 0
        Awl = 0
        Xwl = 0
        p = []
        r = []
        cgTriaMid = []
        a = self.getXwl(h)
        for fh in fvs:  # facet handle
            p.clear()
            r.clear()
            cgTriaMid.clear()
            lpowl.clear()
            lpbwl.clear()
            i = 0
            for vh in fh:  # vertex handle
                p.append(points[vh])
                if p[i][2] > h:
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i = i + 1
            if len(lpowl) < 1:
                # A
                #Ax = points[fvs[fh][0]][0]
                Ax = p[0][0]
                Ay = p[0][1]
                Az = p[0][2]
                # B
                Bx = p[1][0]
                By = p[1][1]
                Bz = p[1][2]
                # C
                Cx = p[2][0]
                Cy = p[2][1]
                Cz = p[2][2]
                areaXYPlane = self.calcArea2DTria(Ax, Ay, Bx, By, Cx, Cy)

                # Volume
                hA = h - Az
                hB = h - Bz
                hC = h - Cz
                vol = vol + 1 / 3 * abs(areaXYPlane) * (hA + hB + hC)

                # Ib Il
                r.append(self.TezisteTrokuta(Ax, Ay, h, Bx, By, h, Cx, Cy, h))
                Ib = Ib + r[0][1] ** 2 * areaXYPlane

                #Kbz, KBx
                hsr = (Az + Bz + Cz) / 3
                cgTriaMid.append(self.TezisteTrokuta(Ax, Ay, (h - hsr), Bx, By, (h - hsr), Cx, Cy, (h - hsr)))
                KBz = KBz + abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))) * (h - hsr) * (
                            hsr + (h - hsr) / 2)
                KBx = KBx + abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))) * (h - hsr) * (
                cgTriaMid[0][0])

                #Awl
                area = self.calcArea2DTria(Ax,Ay,Bx,By,Cx,Cy)

                #Xwl
                Xwl = Xwl + (r[0][0] * abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))))

                pass
            elif len(lpowl) == 1:
                # 2 trokuta Ib, Il i Xwl
                lip = self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]], h, 2)
                r.append(self.TezisteTrokuta(lip[0][0],lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h))
                Ib = Ib + r[0][1] ** 2 * self.calcArea2DTria(lip[0][0],lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])

                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])

                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                Ib = Ib + r[1][1] ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1],p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                Xwl = Xwl + r[1][0] * self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                #Kbz, KBx
                hsr = (lip[0][2] + lip[1][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[1]][0],p[lpbwl[1]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[1]][0],p[lpbwl[1]][1])* (h - hsr) * (cgTriaMid[0][0])

                hsr = (lip[0][2] + p[lpbwl[1]][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1]) * (h - hsr) * (cgTriaMid[0][0])

                #Volume
                area1 = self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[0]][0], p[lpbwl[0]][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                area2 = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                areaXYPlane = area1 + area2
                vol = vol + 1 / 3 * (h - p[lpbwl[0]][2] + h - p[lpbwl[1]][2]) * area1
                vol = vol + 1 / 3 * (h - p[lpbwl[1]][2]) * area2

                # Awl
                area = area1 + area2

                pass
            elif len(lpowl) == 2:
                # 1 trokut Ib, Il
                lip = self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]], h, 2)
                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                Ib = Ib + r[0][1] ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                #KBz, KBx
                hsr = (lip[0][2] + lip[1][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1]) * (h - hsr) * (cgTriaMid[0][0])

                #Volume
                areaXYPlane = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])
                vol = vol + 1 / 3 * (h - p[lpbwl[0]][2]) * areaXYPlane

                #Awl
                area = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                #Xwl
                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                pass
            else:
                area = 0
            Awl = Awl + area

        Xwl = Xwl / Awl
        Il = self.getIl(h, Xwl)
        KBz = KBz / vol
        KBx = KBx / vol
        Lwl,Bwl = self.getLwlBwl(h,fvs,points)
        mfarea = self.getMainFrameArea(x,h,fvs,points)

        return h, vol, Awl, Xwl, KBz, KBx, Ib, Il, Lwl, Bwl, mfarea

    def getIl(self,h, Xwl):
        mesh = self.mesh
        lpowl = []
        lpbwl = []
        Il = 0
        p = []
        r = []
        a = Xwl
        for fh in mesh.faces():  # facet handle
            p.clear()
            r.clear()
            lpowl.clear()
            lpbwl.clear()
            i = 0
            for vh in mesh.fv(fh):  # vertex handle
                p.append(mesh.point(vh))
                if p[i][2] > h:
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i = i + 1
            if len(lpowl) < 1:
                # A
                Ax = p[0][0]
                Ay = p[0][1]
                # B
                Bx = p[1][0]
                By = p[1][1]
                # C
                Cx = p[2][0]
                Cy = p[2][1]
                r.append(self.TezisteTrokuta(Ax, Ay, h, Bx, By, h, Cx, Cy, h))
                Il = Il + (r[0][0] - a) ** 2 * self.calcArea2DTria(Ax, Ay, Bx, By, Cx, Cy)
                pass
            elif len(lpowl) == 1:
                # 2 trokuta
                lip = self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]], h, 2)
                r.append(self.TezisteTrokuta(lip[0][0],lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h))
                Il = Il + (r[0][0] - a) ** 2 * self.calcArea2DTria(lip[0][0],lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])

                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                Il = Il + (r[1][0] - a) ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1],p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1])

                pass
            elif len(lpowl) == 2:
                # 1 trokut
                lip = self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]], h, 2)
                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                Il = Il + (r[0][0] - a) ** 2 * self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])
                pass


        return Il

    def getKBzKBx(self, h):
        mesh = self.mesh
        lpowl=[]
        KBx=0
        KBz=0
        lpbwl = []
        p = []
        cgTriaMid = []
        for fh in mesh.faces():  # facet handle
            p.clear()
            lpowl.clear()
            lpbwl.clear()

            cgTriaMid.clear()
            i=0
            for vh in mesh.fv(fh):  # vertex handle
                p.append(mesh.point(vh))
                if p[i][2] > h :
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i=i+1
            if len(lpowl) < 1:
                # A
                Ax = p[0][0]
                Ay = p[0][1]
                Az = p[0][2]
                # B
                Bx = p[1][0]
                By = p[1][1]
                Bz = p[1][2]
                # C
                Cx = p[2][0]
                Cy = p[2][1]
                Cz = p[2][2]

                hsr = (Az + Bz + Cz) / 3
                cgTriaMid.append(self.TezisteTrokuta(Ax, Ay, (h - hsr), Bx, By, (h - hsr), Cx, Cy, (h - hsr)))
                KBz = KBz + abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))) * (h - hsr) * (cgTriaMid[0][0])

                pass
            elif len(lpowl) ==1:
                # 2 trokuta
                lip=self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]], h,2)
                hsr = (lip[0][2] + lip[1][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[1]][0],p[lpbwl[1]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[1]][0],p[lpbwl[1]][1])* (h - hsr) * (cgTriaMid[0][0])

                hsr = (lip[0][2] + p[lpbwl[1]][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0], p[lpbwl[1]][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[1]][0], p[lpbwl[1]][1], p[lpbwl[0]][0], p[lpbwl[0]][1]) * (h - hsr) * (cgTriaMid[0][0])
                pass
            elif len(lpowl) ==2:
                # 1 trokut
                lip=self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]],h,2)
                hsr = (lip[0][2] + lip[1][2] + p[lpbwl[0]][2]) / 3
                cgTriaMid.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h,lip[1][0], lip[1][1], h, p[lpbwl[0]][0],p[lpbwl[0]][1], h))
                KBz = KBz + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],p[lpbwl[0]][1]) * (h - hsr) * (hsr + (h - hsr) / 2)
                KBx = KBx + self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[0]][0],p[lpbwl[0]][1]) * (h - hsr) * (cgTriaMid[0][0])
                pass

        KBz = KBz / self.getVolume(h)
        KBx = KBx / self.getVolume(h)
        return KBz, KBx

    def TezisteTrokuta(self, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz):

        Xcm = (Ax + Bx + Cx) / 3
        Ycm = (Ay + By + Cy) / 3
        Zcm = (Az + Bz + Cz) / 3

        return Xcm, Ycm, Zcm

    def getXwl(self, h):
        mesh = self.mesh
        lpowl=[]
        Xwl= 0
        lpbwl = []
        p = []
        for fh in mesh.faces():  # facet handle
            p.clear()
            lpowl.clear()
            lpbwl.clear()
            r = []
            r.clear()
            i=0
            for vh in mesh.fv(fh):  # vertex handle
                p.append(mesh.point(vh))
                if p[i][2] > h :
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i=i+1
            if len(lpowl) < 1:
                # A
                Ax = p[0][0]
                Ay = p[0][1]
                # B
                Bx = p[1][0]
                By = p[1][1]
                # C
                Cx = p[2][0]
                Cy = p[2][1]

                r.append(self.TezisteTrokuta(Ax, Ay, h, Bx, By, h, Cx, Cy, h))
                Xwl = Xwl + (r[0][0] * abs(1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))))

                pass
            elif len(lpowl) ==1:
                # 2 trokuta
                lip=self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]],h,2)
                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, lip[1][0], lip[1][1], h,p[lpbwl[1]][0] , p[lpbwl[1]][1], h))
                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[1]][0],p[lpbwl[1]][1])

                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h, p[lpbwl[1]][0] , p[lpbwl[1]][1], h, p[lpbwl[0]][0], p[lpbwl[0]][1], h))
                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0],lip[0][1] ,p[lpbwl[1]][0],p[lpbwl[1]][1], p[lpbwl[0]][0],p[lpbwl[0]][1])
                pass
            elif len(lpowl) ==2:
                # 1 trokut
                lip=self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]],h,2)
                r.append(self.TezisteTrokuta(lip[0][0], lip[0][1], h,lip[1][0], lip[1][1], h, p[lpbwl[0]][0],p[lpbwl[0]][1], h))
                Xwl = Xwl + r[0][0] * self.calcArea2DTria(lip[0][0], lip[0][1],lip[1][0], lip[1][1], p[lpbwl[0]][0], p[lpbwl[0]][1])
                pass
        Xwl = Xwl / self.getAwl(h)
        return Xwl

    def getIntersectionPoints(self, p1,p2,p3,h, os):
        ip1 = self.getIntersectionPoint(p1,p2,h, os)
        ip2 = self.getIntersectionPoint(p1, p3, h, os)
        ips = [ip1, ip2]
        return ips

    def getIntersectionPoint(self, p1,p2,h, os):
        ip1=0
        if os == 2:
            ip1 = [(h-p2[2])/(p1[2]-p2[2])*(p1[0]-p2[0])+p2[0], (h-p2[2])/(p1[2]-p2[2])*(p1[1]-p2[1])+p2[1] ,h]
        if os == 0:
            ip1 = [h, (h-p1[0])/(p2[0]-p1[0])*(p2[1]-p1[1])+p1[1], (h-p1[0])/(p2[0]-p1[0])*(p2[2]-p1[2])+p1[2]]

        return ip1

    def getAwl(self, hvl):
        mesh = self.mesh
        h = hvl
        Awl = 0
        lpowl=[]
        lpbwl = []
        p = []
        for fh in mesh.faces():  # facet handle
            p.clear()
            lpowl.clear()
            lpbwl.clear()
            i=0
            for vh in mesh.fv(fh):  # vertex handle
                p.append(mesh.point(vh))
                if p[i][2] > h :
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i=i+1
            if len(lpowl) < 1:
                # A
                Ax = p[0][0]
                Ay = p[0][1]
                # B
                Bx = p[1][0]
                By = p[1][1]
                # C
                Cx = p[2][0]
                Cy = p[2][1]
                area = self.calcArea2DTria(Ax,Ay,Bx,By,Cx,Cy)
                pass
            elif len(lpowl) ==1:
                # 2 trokuta
                lip=self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]],h,2)
                area1 = self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[0]][0], p[lpbwl[0]][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                area2 = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                area = area1+area2
                pass
            elif len(lpowl) ==2:
                # 1 trokut
                lip=self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]],h,2)
                area = self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[0]][0],p[lpbwl[0]][1])
                pass
            else:
                area = 0
            Awl = Awl + area
        return Awl

    def calcArea2DTria(self,Ax,Ay,Bx,By,Cx,Cy):
        area = 1 / 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))
        return abs(area)

    def calcAreaTria2Dxy(self,p1,p2,p3):
        return self.calcAreaTria3D(p1[0:2],p2[0:2],p3[0:2])

    def calcAreaTria3D(self,p1,p2,p3):
        p1p2= np.subtract(p2,p1)
        p1p3 = np.subtract(p3, p1)
        u=np.cross(p1p2,p1p3)
        return np.linalg.norm(u)/2

    def getVolume(self, h):
        mesh = self.mesh
        vol =0
        lpowl=[]
        lpbwl = []
        p = []
        for fh in mesh.faces():  # facet handle
            p.clear()
            lpowl.clear()
            lpbwl.clear()
            i=0
            for vh in mesh.fv(fh):  # vertex handle
                p.append(mesh.point(vh))
                if p[i][2] > h :
                    lpowl.append(i)

                else:
                    lpbwl.append(i)
                i=i+1
            if len(lpowl) < 1:
                # A
                Ax = p[0][0]
                Ay = p[0][1]
                # B
                Bx = p[1][0]
                By = p[1][1]
                # C
                Cx = p[2][0]
                Cy = p[2][1]
                areaXYPlane = self.calcArea2DTria(Ax,Ay,Bx,By,Cx,Cy)
                Az = p[0][2]
                Bz = p[1][2]
                Cz = p[2][2]
                hA = h - Az
                hB = h - Bz
                hC = h - Cz
                vol = vol + 1/3* abs(areaXYPlane) * (hA + hB + hC)
                pass
            elif len(lpowl) ==1:
                # 2 trokuta
                lip=self.getIntersectionPoints(p[lpowl[0]], p[lpbwl[0]], p[lpbwl[1]],h,2)
                area1 = self.calcArea2DTria(lip[0][0], lip[0][1], p[lpbwl[0]][0], p[lpbwl[0]][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                area2 = self.calcArea2DTria(lip[0][0], lip[0][1], lip[1][0], lip[1][1], p[lpbwl[1]][0], p[lpbwl[1]][1])
                areaXYPlane = area1+area2
                vol = vol + 1/3*(h-p[lpbwl[0]][2]+h-p[lpbwl[1]][2])*area1
                vol = vol + 1 / 3 * (h - p[lpbwl[1]][2]) * area2
                pass
            elif len(lpowl) ==2:
                # 1 trokut
                lip=self.getIntersectionPoints(p[lpbwl[0]], p[lpowl[0]], p[lpowl[1]],h,2)
                areaXYPlane = self.calcArea2DTria(lip[0][0],lip[0][1] ,lip[1][0],lip[1][1], p[lpbwl[0]][0],p[lpbwl[0]][1])
                vol = vol + 1/3*(h-p[lpbwl[0]][2])*areaXYPlane
                pass
            else:
                areaXYPlane = 0
        return vol


    # kod za formu
    def genHullFormMesh(self, lines: list):
        mesh = om.TriMesh()
        wlinesPos = lines[0]  # positive y waterlines
        wlinesNeg = lines[1]  # negative y waerlines
        wlKeel = lines[2]  # keel waterline (one waterline)
        n1 = np.array([0,0,0])
        m1 = np.array([0,0,0])
        m2 = np.array([0,0,0])
        pt1= np.array(3)
        pt2= np.array(3)
        n2 = np.array([0,0,0])
        n3 = np.array([0,0,0])
        n4 = np.array([0,0,0])

        for i in range(len(lines) - 2):  # lijevo desno kobilica
            for j in range(len(lines[i])):  # broji vodne linije
                for k in range(len(lines[i][j]) - 2):  # broj tocaka na vodnoj liniji
                    if j == len(lines[i]) - 1:          #kobilica
                        mpt1 = lines[i][j-1][k]
                        mpt2 = lines[i][j-1][k + 1]
                        mpt3 = lines[i][len(lines[i])-1][k]
                        mpt4 = lines[i][len(lines[i])-1][k + 1]

                        # volumenA = volumenA + Voltot(mpt1,mpt2,mpt3,mpt4)
                        mesh.add_face(mesh.add_vertex(mpt1), mesh.add_vertex(mpt2), mesh.add_vertex(mpt3))
                        mesh.add_face(mesh.add_vertex(mpt2), mesh.add_vertex(mpt4), mesh.add_vertex(mpt3))

                    if j != len(lines[i])-1:         #Sve ostale vodne linije
                        mpt1 = lines[i][j][k]
                        mpt2 = lines[i][j][k + 1]
                        mpt3 = lines[i][j+1][k]
                        mpt4 = lines[i][j+1][k + 1]
                        mesh.add_face(mesh.add_vertex(mpt1), mesh.add_vertex(mpt2), mesh.add_vertex(mpt3))
                        mesh.add_face(mesh.add_vertex(mpt2), mesh.add_vertex(mpt4), mesh.add_vertex(mpt3))


        for i in range(len(lines) - 2):  # lijevo desno kobilica
                for k in range(len(lines[i][0]) - 2):  # broj tocaka na vodnoj liniji
                        pt1 = lines[i][0][k]
                        pt2 = lines[i][0][k + 1]
                        m1[0] = pt1[0]
                        m1[2] = pt1[2]
                        m2[0] = pt2[0]

                        m2[2] = pt2[2]
                        mesh.add_face(mesh.add_vertex(pt1), mesh.add_vertex(m1), mesh.add_vertex(pt2))
                        mesh.add_face(mesh.add_vertex(pt2), mesh.add_vertex(m1), mesh.add_vertex(m2))

        return mesh

    def _genFaces(self,mesh:om.TriMesh,whs:list, doReverse:bool):
        nl=len(whs)
        npt=len(whs[0])
        for iL in range(1, nl):
            npt_iL = len(whs[iL])
            npt_iL_1 = len(whs[iL-1])
            dip=0
            if npt_iL > npt_iL_1:
                if doReverse:
                    mesh.add_face(whs[iL][0], whs[iL][1], whs[iL - 1][0])
                else:
                    mesh.add_face(whs[iL][1], whs[iL][0], whs[iL - 1][0])
                dip = 1
            for ipL_1 in range(1,npt_iL_1):
                ip = ipL_1+dip
                if doReverse:
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip], whs[iL - 1][ipL_1])
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip - 1], whs[iL][ip])
                else:
                    mesh.add_face(whs[iL - 1][ipL_1-1],   whs[iL - 1][ipL_1],whs[iL ][ip])
                    mesh.add_face(whs[iL - 1][ipL_1 - 1], whs[iL][ip],    whs[iL][ip-1])

    def genHullFormMeshPP(self, lines: list):
        mesh = om.TriMesh()
        wlinesPos = lines[0]  # positive y waterlines
        wlinesNeg = lines[1]  # negative y waerlines
        wlKeel = lines[2]  # keel waterline (one waterline)
        wlinesPos.reverse()
        wlinesNeg.reverse()

        whsPos = []
        whsNeg = []
        whsi = []
        whsPos.append(whsi)
        whsNeg.append(whsi)
        for p in wlKeel:
            whsi.append(mesh.add_vertex(p))


        for wl in wlinesPos:
            whsi = []
            whsPos.append(whsi)
            for p in wl:
                whsi.append(mesh.add_vertex(p))
        for wl in wlinesNeg:
            whsi = []
            whsNeg.append(whsi)
            for p in wl:
                whsi.append(mesh.add_vertex(p))

        self._genFaces(mesh,whsPos,True)
        self._genFaces(mesh, whsNeg,False)

        return mesh



    def hullGen(self, shipdata: dict, pdecks: list, nump):
        # gs is the grid size of a cell, in pixels
        # Reminder to make gridsize scaled to the screen width
        # Sets hullformdir data to slider values
        shipdata["loa_val"] = shipdata["loa_val"]
        shipdata["boa_val"] = shipdata["boa_val"]

        #
        midshipsM = shipdata["ms_val"]  # Constant m in JC equation
        bowRakeM = shipdata["bow_val"]  # Constant m in JC equation
        transomM = shipdata["tr_val"]  # Constant m in JC equation
        fwdDeckM = shipdata["deck_val"]  # Constant m in JC equation

        transomBeamMax = (shipdata["boa_val"] * shipdata["tb_val"]) / 2  # Transom half beam
        transomTip = shipdata["draft_val"] * shipdata["td_val"]
        ACU = shipdata["loa_val"] * shipdata["acu_val"]
        keelFwd = shipdata["loa_val"] * shipdata["kf_val"]
        slope = shipdata["sa_val"]

        midBeam = []  # Array with midships half beam per deck
        bowRake = []  # Array with bow rake per deck
        bowRakeS = []  # Array with bow rake per deck in superstructure
        TB = 0  # Transom half beam of a deck
        transomBeam = []  # Array with transom half beam per deck
        fwdDeckMArray = []  # Array with constants m in JC equation for deck outlines
        AE = 0  # Aft end of a deck
        aftEnd = []  # Array with aft end of each deck
        aftEndS = []  # Array with aft end of each deck in superstructure
        noseConeBaseRadius = []  # See excel tool
        ogiveRadius = []  # See excel tool
        pdecks2 = []  # Array with deck positions of hull decks
        pdecks3 = []  # Array with deck positions of superstructure decks

        for i in range(len(pdecks)):  # Assign values to variables above
            if pdecks[i] <= shipdata["draft_val"]:  # For each deck that is in the hull
                midBeam.append((Math.acosh(
                    (pdecks[i] / shipdata["draft_val"]) * (Math.cosh(midshipsM * Math.pi) - 1) + 1) / (
                                            midshipsM * Math.pi)) * (shipdata["boa_val"] / 2))
                bowRake.append((Math.acosh(
                    (pdecks[i] / shipdata["draft_val"]) * (Math.cosh(bowRakeM * Math.pi) - 1) + 1) / (
                                            bowRakeM * Math.pi)) * (shipdata["loa_val"] - keelFwd))
                if pdecks[i] > transomTip:
                    TB = ((Math.acosh(((pdecks[i] - transomTip) / (shipdata["draft_val"] - transomTip)) * (
                                Math.cosh(transomM * Math.pi) - 1) + 1) / (transomM * Math.pi)) * (transomBeamMax))

                else:
                    TB = 0

                transomBeam.append(TB)
                fwdDeckMArray.append(fwdDeckM * (pdecks[i] / (shipdata[
                    "draft_val"])) + 0.001)  # Changes constant m in JC equation to make deck outlines becomes slimmer with decreasing z position (see below)
                if (pdecks[i] >= transomTip):
                    AE = (shipdata["draft_val"] - pdecks[i]) * Math.tan(slope)

                else:
                    AE = (shipdata["draft_val"] - transomTip) * Math.tan(slope) + (transomTip - pdecks[i]) * (
                                (ACU - (shipdata["draft_val"] - transomTip) * Math.tan(slope)) / transomTip)

                aftEnd.append(AE)
                pdecks2.append(pdecks[i])

            else:  # For each deck that is in the superstructure
                aftEndS.append((pdecks[i] - shipdata["draft_val"]) * Math.tan(slope))
                bowRakeS.append(shipdata["loa_val"] - ((pdecks[i] - shipdata["draft_val"]) * Math.tan(slope)) - keelFwd)
                pdecks3.append(pdecks[i])

        for i in range(len(midBeam)):  # Assign values to variables above cont.
            noseConeBaseRadius.append(midBeam[i] - transomBeam[i])
            if noseConeBaseRadius[i] > 0:
                ogiveRadius.append(
                    (Math.pow(noseConeBaseRadius[i], 2) + Math.pow((shipdata["loa_val"] / 2) - aftEnd[i], 2)) / (
                                2 * noseConeBaseRadius[i]))

            else:
                ogiveRadius.append(0)

        deckOutlinesHull = []  # Array with hull deck outline x, y coordinates
        # Get y points for every x
        for idk in range(len(midBeam)):  # For each deck in hull
            deckOutlinesHull.append([])  # For each deck create array
            if pdecks2[idk] != 0:  # If not keel
                if transomBeam[idk] > 0:  # Add vertical hull line at transom
                    deckOutlinesHull[idk].append([aftEnd[idk], 0])
                kmin = aftEnd[idk]
                kmax = shipdata["loa_val"] / 2
                klist = np.linspace(kmin, kmax, nump)
                for xpt in klist:
                    deckOutlinesHull[idk].append([xpt, (
                                Math.sqrt(Math.pow(ogiveRadius[idk], 2) - Math.pow(xpt - shipdata["loa_val"] / 2, 2)) +
                                noseConeBaseRadius[idk] - ogiveRadius[idk] + transomBeam[idk])])

                kmin = shipdata["loa_val"] / 2
                kmax = keelFwd + bowRake[idk]
                klist = np.linspace(kmin, kmax, nump)
                for xpt in klist:
                    eqX = (xpt - shipdata["loa_val"] / 2) / (
                                keelFwd + bowRake[idk] - (shipdata["loa_val"] / 2))  # Value of x in JC equation
                    deckOutlinesHull[idk].append([xpt, (1 - ((Math.cosh(eqX * fwdDeckMArray[idk] * Math.pi) - 1) / (
                                Math.cosh(fwdDeckMArray[idk] * Math.pi) - 1))) * midBeam[idk]])


            else:  # If keel draw top
                kmin = aftEnd[idk]
                kmax = (keelFwd + bowRake[idk])
                klist = np.linspace(kmin, kmax, nump * 2)
                for xpt in klist:
                    deckOutlinesHull[idk].append([xpt, 0])  # Straight line

        deckOutlinesS = []  # Array with superstructure deck outline x, y coordinates
        tumblehome = []  # Superstructure tumblehome
        for n in range(len(aftEndS)):  # For each deck in superstructure
            deckOutlinesS.append([])  # For each deck create array
            tumblehome = (pdecks3[n] - shipdata["draft_val"]) * Math.tan(
                slope)  # Calculate tumblehome y offset to subtract below
            deckOutlinesS[n].append([aftEndS[n], 0])  # Add vertical hull line at transom

            kmin = aftEndS[n]
            kmax = shipdata["loa_val"] / 2
            klist = np.linspace(kmin, kmax, nump)
            for xpt in klist:
                deckOutlinesS[n].append([xpt, (
                            Math.sqrt(Math.pow(ogiveRadius[0], 2) - Math.pow(xpt - shipdata["loa_val"] / 2, 2)) +
                            noseConeBaseRadius[0] - ogiveRadius[0] + transomBeam[0] - tumblehome)])

            kmin = shipdata["loa_val"] / 2
            kmax = (keelFwd + bowRakeS[n])
            klist = np.linspace(kmin, kmax, nump)
            for xpt in klist:
                eqX = (xpt - shipdata["loa_val"] / 2) / (
                            keelFwd + bowRakeS[n] - (shipdata["loa_val"] / 2))  # Value of x in JC equation
                deckOutlinesS[n].append([xpt, (1 - ((Math.cosh(eqX * fwdDeckMArray[0] * Math.pi) - 1) / (
                            Math.cosh(fwdDeckMArray[0] * Math.pi) - 1))) * (midBeam[0] - tumblehome)])

        wlinesPos = []
        wlinesNeg = []
        wlKeel = []

        for ii in range(len(deckOutlinesS)):
            wlineP = list()
            wlineN = list()
            for item in deckOutlinesS[ii]:
                p = np.array([item[0], item[1], pdecks3[ii]])
                wlineP.append(p)
                p = np.array([item[0], -item[1], pdecks3[ii]])
                wlineN.append(p)
            wlinesPos.append(wlineP)
            wlinesNeg.append(wlineN)

        for ii in range(len(deckOutlinesHull)):

            if pdecks2[ii] != 0:
                wlineP = list()
                wlineN = list()
                for item in deckOutlinesHull[ii]:
                    p = np.array([item[0], item[1], pdecks2[ii]])
                    wlineP.append(p)
                    p = np.array([item[0], -item[1], pdecks2[ii]])
                    wlineN.append(p)
                wlinesPos.append(wlineP)
                wlinesNeg.append(wlineN)
            else:
                for item in deckOutlinesHull[ii]:
                    p = np.array([item[0], item[1], pdecks2[ii]])
                    wlKeel.append(p)

        return [wlinesPos, wlinesNeg, wlKeel]

    def readShipData(self):
        shipdata = {}
        pdecks = []
        pbulkheads = []
        with open(self.filename, newline='') as csvfile:
            f = csv.DictReader(csvfile)
            shipset = 0
            for row in f:  # there is only one row after header row!!!!!
                shipset = row

            shipdata["loa_val"] = float(shipset['LOA'])  # treba li učitavanje vrijednosti biti u petlji?
            shipdata["boa_val"] = float(shipset['BOA'])  # treba li učitavanje vrijednosti biti u petlji?
            shipdata['draft_val'] = float(shipset['D'])
            shipdata['shipname'] = shipset['Name']

            splitdata = str(shipset['HullData']).split(" ")
            shipdata["ms_val"] = float(splitdata[0])
            shipdata["bow_val"] = float(splitdata[1])
            shipdata["tr_val"] = float(splitdata[2])
            shipdata["deck_val"] = float(splitdata[3])
            shipdata["tb_val"] = float(splitdata[4])
            shipdata["td_val"] = float(splitdata[5])
            shipdata["acu_val"] = float(splitdata[6])
            shipdata["kf_val"] = float(splitdata[7])
            shipdata["sa_val"] = float(splitdata[8])

            shipdata["sp_val"] = float(splitdata[9])
            shipdata["cwl_val"] = float(splitdata[10])
            shipdata["lcb_val"] = float(splitdata[11])
            shipdata["cb_val"] = float(splitdata[12])
            shipdata["mc_val"] = float(splitdata[13])
            shipdata["bb_val"] = float(splitdata[14])
            shipdata["tran_val"] = float(splitdata[15])
            shipdata["ab_val"] = float(splitdata[16])

            shipdata["lwl_val"] = float(splitdata[17])
            shipdata["bwl_val"] = float(splitdata[18])
            shipdata["tf_val"] = float(splitdata[19])
            shipdata["ta_val"] = float(splitdata[20])

            shipdata["app1"] = float(splitdata[21])
            shipdata["area_app1"] = float(splitdata[22])
            shipdata["app2"] = float(splitdata[23])
            shipdata["area_app2"] = float(splitdata[24])
            shipdata["area_app3"] = float(splitdata[25])
            shipdata["area_app4"] = float(splitdata[26])
            shipdata["app5"] = float(splitdata[27])
            shipdata["area_app5"] = float(splitdata[28])
            shipdata["area_app6"] = float(splitdata[29])
            shipdata["area_app7"] = float(splitdata[30])
            shipdata["app8"] = float(splitdata[31])
            shipdata["area_app8"] = float(splitdata[32])
            shipdata["area_app9"] = float(splitdata[33])
            shipdata["area_app10"] = float(splitdata[34])
            shipdata["area_app11"] = float(splitdata[35])

            shipdata["cg_val"] = float(splitdata[36])
            shipdata["heading_val"] = float(splitdata[37])
            shipdata["amplitude_val"] = float(splitdata[38])
            shipdata["roll_val"] = float(splitdata[39])
            shipdata["damping_val"] = float(splitdata[40])
            shipdata["plr_val"] = float(splitdata[41])
            shipdata["gmt_val"] = float(splitdata[42])

            draft = shipdata["draft_val"]
            splitdata = str(shipset['DeckPos']).split(" ")
            for dp in splitdata:
                pdecks.append(float(dp))
            splitdata = str(shipset['BHPos']).split(" ")
            for dp in splitdata:
                pbulkheads.append(float(dp))
        return  shipdata,pdecks,pbulkheads
