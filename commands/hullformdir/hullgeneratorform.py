import numpy as np

from hullformdir.hullform import *
import csv
import math as Math
import time
#d3v imports
from iohandlers import IOHandler
from signals import Signals


from typing import Set,Dict,List

# import matplotlib.pyplot as plt
def writecsv_dictionary(filepath: str, dict: Dict):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict)
def get_perpendicular_offset_point(p0:np.ndarray,p1:np.ndarray,plane_normal:np.ndarray, offset):
    v1=p1-p0
    uvec= np.cross(v1,plane_normal)
    uvec = uvec/np.linalg.norm(uvec)
    perp_point= p0+uvec*offset
    return perp_point

def get_point_intersection_two_lines(a0:np.ndarray,a1:np.ndarray,b0:np.ndarray,b1:np.ndarray):
    res = np.linalg.lstsq(np.array([a1 - a0, b0 - b1]).T, b0 - a0)
    t=res[0][0]
    p = (1 - t) * a0 + t * a1
    #p = (1 - s) * b0 + s * b1
    return  p

class HullGenFormMeshControl:
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

    def genWLPositions(self,hWL_top,  hWL_bottom,hst=None ):
        if hst is not None:
            wlPos = self._getDistribution(hWL_top, hWL_bottom, hst,1)
        else:
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


class HullGeneratorForm(HullForm):
    def __init__(self, fileName,name="",translation=np.zeros(3)):
        super().__init__(fileName, name,translation)
        self.shipdata = {}
        self.pdecks =[]
        self.dict_decks = {}
        self.pbulkheads = []
        self.hfmq = HullGenFormMeshControl()
        filename_no_ext, file_extension = os.path.splitext(fileName)
        if file_extension == ".huf":
            self.shipdata,self.pdecks,self.pbulkheads,self.dict_decks = self.readUCLDBBShipData()
            self._filename=filename_no_ext+".hgf"
            self.write_file(self.filename)
        elif file_extension == ".hgf":
            self.shipdata, self.pdecks, self.pbulkheads, self.dict_decks = self.read_file()
        self.mesh,self.wlinesPos,self.wlinesNeg,self.wlKeel = self._generateHullForm()
        self.rise_stern()


    def regenerateHullHorm(self):
        self.mesh, self.wlinesPos, self.wlinesNeg, self.wlKeel = self._generateHullForm()
        self.rise_stern()

    def rise_stern(self):
        use_rise = np.abs(1.0 - np.abs(self.shipdata['anc_rise_bow']*self.shipdata['anc_rise_stern'])) < 0.001
        if use_rise:
            depth = self.shipdata["depth_val"]
            rise_bow = depth * self.shipdata['anc_rise_bow']
            rise_stern = depth * self.shipdata['anc_rise_stern']
            self.rise_mesh_ends(rise_bow,rise_stern)
    def _generateHullForm(self):
        transomTip = self.shipdata["depth_val"] * self.shipdata["td_val"]
        obligatoryWL= []
        for dh in self.pdecks:
            obligatoryWL.append(dh)
        obligatoryWL.append(transomTip)
        obligatoryWL.sort(reverse=True)
        wlPos = self.hfmq.genWLPositionsUsingObligatory(obligatoryWL)
        lines = self.hullGen(self.shipdata, wlPos, self.hfmq.numPointsWLhalf)
        #lines[0]  +y waterlines;lines[1]  -y waerlines; lines[2]  # keel waterline (one)
        mesh = self._generateMesh(lines)
        return mesh, lines[0],lines[1],lines[2]

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

    def _generateMesh(self, lines: list):
        wooden_keel_width = self.shipdata['anc_keel_width']
        wooden_keel_height = self.shipdata['anc_keel_height']
        use_wooden_keel = (wooden_keel_width*wooden_keel_height > 0.0)
        mesh = om.TriMesh()
        om.PolyMesh()
        wlinesPos = lines[0]  # positive y waterlines
        wlinesNeg = lines[1]  # negative y waerlines
        wlKeel = lines[2]  # keel waterline (one waterline)
        wlinesPos.reverse()
        wlinesNeg.reverse()

        whsPos = []
        whsNeg = []
        if use_wooden_keel:
            loa = self.shipdata["loa_val"]
            # determine wooden keel stern vector
            a0=wlinesPos[-1][0].copy()
            keel_stern_vector=a0-wlinesPos[-2][0]
            keel_stern_vector = keel_stern_vector/np.linalg.norm(keel_stern_vector)
            a0[0] = a0[0]- wooden_keel_height
            a1= a0+keel_stern_vector*self.shipdata["depth_val"]*2
            # determine wooden keel bow vectors
            wl_bow_offset_points=[]
            wl_last= wlKeel
            plane_normal = np.array([0.0,1.0,0.0])
            for i in range(len(wlinesPos)):
                wl=wlinesPos[i]
                bow_a0 = wl[-1].copy()
                bow_a1 = wl_last[-1].copy()
                bow_p = get_perpendicular_offset_point(bow_a0,bow_a1,plane_normal,wooden_keel_height)
                wl_bow_offset_points.append(bow_p)
                wl_last = wl

            num_pts=len(wlKeel)
            wlKeel_np= np.array(wlKeel)
            wlKeel_cl_offset =wlKeel_np.copy()
            wlKeel_cl_offset[:, 2] = wlKeel_cl_offset[:, 2] - wooden_keel_height
            # Keel -  Positive waterlines
            wlKeel_pos = wlKeel_np.copy()
            wlKeel_pos[:,1] = wooden_keel_width
            wlKeel_pos_offset = wlKeel_pos.copy()
            wlKeel_pos_offset[:, 2] = wlKeel_pos_offset[:, 2]-wooden_keel_height
            #WL 1 - simmetry line, offset down
            whsi = []
            whsPos.append(whsi)
            p=get_point_intersection_two_lines(a0,a1,wlKeel_cl_offset[0, :],wlKeel_cl_offset[0, :]+np.array([-wooden_keel_height*5,0.0,0.0]))
            whsi.append(mesh.add_vertex(p))
            for i in range(num_pts):
                pi=wlKeel_cl_offset[i,:]
                whsi.append(mesh.add_vertex(pi))
            p_1 = get_point_intersection_two_lines(wl_bow_offset_points[0], wl_bow_offset_points[1],
                                                   np.array([0.0, 0.0, p[2]]),
                                                   np.array([loa, 0.0, p[2]]))
            whsi.append(mesh.add_vertex(p_1))
            # WL 2 - keel width, offset down
            whsi = []
            whsPos.append(whsi)
            p1 =  p.copy()
            p0 = p.copy()
            p1[1] = wooden_keel_width
            whsi.append(mesh.add_vertex(p0))
            whsi.append(mesh.add_vertex(p1))
            for i in range(num_pts):
                pi = wlKeel_pos_offset[i, :]
                whsi.append(mesh.add_vertex(pi))
            p_1 = get_point_intersection_two_lines(wl_bow_offset_points[0], wl_bow_offset_points[1], np.array([0.0, 0.0, p0[2]]),
                                                  np.array([loa, 0.0, p0[2]]))
            p_2 = p_1.copy()
            p_2[1] = wooden_keel_width
            whsi.append(mesh.add_vertex(p_2))
            whsi.append(mesh.add_vertex(p_1))
            # WL 3 - keel width, on hull
            whsi = []
            whsPos.append(whsi)
            p1 = wlKeel_pos[0, :].copy()
            p1[0] = 0
            p0 = get_point_intersection_two_lines(a0, a1, p1,
                                                 p1 + np.array([-wooden_keel_height * 5, 0.0, 0.0]))
            p1 = p0.copy()
            p1[1] = wooden_keel_width
            whsi.append(mesh.add_vertex(p0))
            whsi.append(mesh.add_vertex(p1))
            wl_keel_pos = []
            wl_keel_pos.append(p0)
            wl_keel_pos.append(p)
            for i in range(num_pts):
                pi = wlKeel_pos[i, :]
                whsi.append(mesh.add_vertex(pi))
                wl_keel_pos.append(pi)
            p_1 = wl_bow_offset_points[0]
            p_2 = p_1.copy()
            p_2[1] = wooden_keel_width
            whsi.append(mesh.add_vertex(p_2))
            whsi.append(mesh.add_vertex(p_1))
            # Keel -  Negative waterlines
            wlKeel_neg = wlKeel_np.copy()
            wlKeel_neg[:, 1] = -wooden_keel_width
            wlKeel_neg_offset = wlKeel_neg.copy()
            wlKeel_neg_offset[:, 2] = wlKeel_neg_offset[:, 2] - wooden_keel_height
            # WL 1 - simmetry line, offset down
            whsi = []
            whsNeg.append(whsi)
            p = get_point_intersection_two_lines(a0, a1, wlKeel_cl_offset[0, :],
                                                 wlKeel_cl_offset[0, :] + np.array([-wooden_keel_height * 5, 0.0, 0.0]))
            whsi.append(mesh.add_vertex(p))
            for i in range(num_pts):
                pi = wlKeel_cl_offset[i, :]
                whsi.append(mesh.add_vertex(pi))
            p_1 = get_point_intersection_two_lines(wl_bow_offset_points[0], wl_bow_offset_points[1], np.array([0.0, 0.0, p[2]]),
                                                  np.array([loa, 0.0, p[2]]))
            whsi.append(mesh.add_vertex(p_1))
            # WL 2 - keel width, offset down
            whsi = []
            whsNeg.append(whsi)
            p1 = p.copy()
            p0 = p.copy()
            p1[1] = -wooden_keel_width
            whsi.append(mesh.add_vertex(p0))
            whsi.append(mesh.add_vertex(p1))
            for i in range(num_pts):
                pi = wlKeel_neg_offset[i, :]
                whsi.append(mesh.add_vertex(pi))

            p_2 = p_1.copy()
            p_2[1] = -wooden_keel_width
            whsi.append(mesh.add_vertex(p_2))
            whsi.append(mesh.add_vertex(p_1))
            # WL 3 - keel width, on hull
            whsi = []
            whsNeg.append(whsi)
            p1 = wlKeel_neg[0, :].copy()
            p1[0] = 0
            p0 = get_point_intersection_two_lines(a0, a1, p1,
                                                  p1 + np.array([-wooden_keel_height * 5, 0.0, 0.0]))
            p1=p0.copy()
            p1[1] = -wooden_keel_width
            whsi.append(mesh.add_vertex(p0))
            whsi.append(mesh.add_vertex(p1))
            wl_keel_neg = []
            wl_keel_neg.append(p0)
            wl_keel_neg.append(p1)
            for i in range(num_pts):
                pi = wlKeel_neg[i, :]
                whsi.append(mesh.add_vertex(pi))
                wl_keel_neg.append(pi)
            p_1 = wl_bow_offset_points[0]
            p_2 = p_1.copy()
            p_2[1] = -wooden_keel_width
            whsi.append(mesh.add_vertex(p_2))
            whsi.append(mesh.add_vertex(p_1))
            # Positive waterlines
            for i in range(len(wlinesPos)):
                wl=wlinesPos[i]
                whsi = []
                whsPos.append(whsi)
                p0 = get_point_intersection_two_lines(a0, a1, wl[0],
                                                      wl[0] + np.array([-wooden_keel_height * 5, 0.0, 0.0]))
                whsi.append(mesh.add_vertex(p0))
                p1 = p0.copy()
                p1[1] =  wooden_keel_width
                whsi.append(mesh.add_vertex(p1))
                for ip in range(len(wl)):
                    p=wl[ip]
                    if p[1]<wooden_keel_width:
                        p[1] = wooden_keel_width
                    whsi.append(mesh.add_vertex(p))
                p_1=wl_bow_offset_points[i]
                p_2 =p_1.copy()
                p_2[1] = wooden_keel_width
                whsi.append(mesh.add_vertex(p_2))
                whsi.append(mesh.add_vertex(p_1))
            # Negative waterlines

            for i in range(len(wlinesNeg)):
                wl=wlinesNeg[i]
                whsi = []
                whsNeg.append(whsi)
                p0 = get_point_intersection_two_lines(a0, a1, wl[0],
                                                      wl[0] + np.array([-wooden_keel_height * 5, 0.0, 0.0]))
                whsi.append(mesh.add_vertex(p0))
                p1 = p0.copy()
                p1[1] =  - wooden_keel_width
                whsi.append(mesh.add_vertex(p1))
                for ip in range(len(wl)):
                    p = wl[ip]
                    if p[1] > -wooden_keel_width:
                        p[1] = -wooden_keel_width
                    whsi.append(mesh.add_vertex(p))
                p_1 = wl_bow_offset_points[i]
                p_2 = p_1.copy()
                p_2[1] = -wooden_keel_width
                whsi.append(mesh.add_vertex(p_2))
                whsi.append(mesh.add_vertex(p_1))
        else:
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

    def hullGen(self, shipdata: dict, pdecks: list, nump,frame_positions:List[float] = None):
        # gs is the grid size of a cell, in pixels
        # Reminder to make gridsize scaled to the screen width

        use_forward_model_for_aft= shipdata['use_fwd_for_aft']
        loa = shipdata["loa_val"]
        boa = shipdata["boa_val"]
        depth = shipdata["depth_val"]

        #
        midshipsM = shipdata["ms_val"]  # Constant m in JC equation
        bowRakeM = shipdata["bow_val"]  # Constant m in JC equation
        transomM = shipdata["tr_stern_val"]  # Constant m in JC equation
        fwdDeckM = shipdata["deck_val"]  # Constant m in JC equation

        transomBeamMax = (boa * shipdata["tb_val"]) / 2  # Transom half beam
        transomTip = depth * shipdata["td_val"]
        ACU = loa * shipdata["acu_val"]
        keelFwd = loa * shipdata["kf_val"]
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
        if use_forward_model_for_aft:
            stern_rakeM = shipdata["tr_stern_val"]
            keel_stern = loa * shipdata["acu_val"]
            sternRake = []  # Array with stern rake per deck
            stern_deck_fullness = shipdata["tb_val"]
            sternDeckMArray = []
        for i in range(len(pdecks)):  # Assign values to variables above
            if pdecks[i] <= depth:  # For each deck that is in the hull
                midBeam.append((Math.acosh(
                    (pdecks[i] / depth) * (Math.cosh(midshipsM * Math.pi) - 1) + 1) / (
                                            midshipsM * Math.pi)) * (boa / 2))
                bowRake.append((Math.acosh(
                    (pdecks[i] / depth) * (Math.cosh(bowRakeM * Math.pi) - 1) + 1) / (
                                            bowRakeM * Math.pi)) * (loa - keelFwd))
                if use_forward_model_for_aft:
                    sternRake.append((Math.acosh(
                        (pdecks[i] / depth) * (Math.cosh(stern_rakeM * Math.pi) - 1) + 1) / (
                                            stern_rakeM * Math.pi)) * (keel_stern))

                if pdecks[i] > transomTip:
                    TB = ((Math.acosh(((pdecks[i] - transomTip) / (depth - transomTip)) * (
                                Math.cosh(transomM * Math.pi) - 1) + 1) / (transomM * Math.pi)) * (transomBeamMax))

                else:
                    TB = 0

                transomBeam.append(TB)
                fwdDeckMArray.append(fwdDeckM * (pdecks[i] / depth) + 0.001)  # Changes constant m in JC equation to make deck outlines becomes slimmer with decreasing z position (see below)
                if use_forward_model_for_aft:
                    sternDeckMArray.append(stern_deck_fullness * (pdecks[i] / depth) + 0.001)
                    AE= keel_stern - ((Math.acosh(
                        (pdecks[i] / depth) * (Math.cosh(stern_rakeM * Math.pi) - 1) + 1) / (
                                            stern_rakeM * Math.pi)) * (keel_stern))
                    AE1 = loa - ((Math.acosh(
                        (pdecks[i] / depth) * (Math.cosh(bowRakeM * Math.pi) - 1) + 1) / (
                                         bowRakeM * Math.pi)) * (loa - keelFwd) + keelFwd)
                    diffAE=AE-AE1
                    tmpaaa=10
                else:
                    if (pdecks[i] >= transomTip):
                        AE = (depth - pdecks[i]) * Math.tan(slope)

                    else:
                        AE = (depth - transomTip) * Math.tan(slope) + (transomTip - pdecks[i]) * (
                                    (ACU - (depth - transomTip) * Math.tan(slope)) / transomTip)

                aftEnd.append(AE)
                pdecks2.append(pdecks[i])


            else:  # For each deck that is in the superstructure
                aftEndS.append((pdecks[i] - depth) * Math.tan(slope))
                bowRakeS.append(loa - ((pdecks[i] - depth) * Math.tan(slope)) - keelFwd)
                pdecks3.append(pdecks[i])


        for i in range(len(midBeam)):  # Assign values to variables above cont.
            noseConeBaseRadius.append(midBeam[i] - transomBeam[i])
            if noseConeBaseRadius[i] > 0:
                ogiveRadius.append(
                    (Math.pow(noseConeBaseRadius[i], 2) + Math.pow((loa / 2) - aftEnd[i], 2)) / (
                                2 * noseConeBaseRadius[i]))

            else:
                ogiveRadius.append(0)

        deckOutlinesHull = []  # Array with hull deck outline x, y coordinates
        # Get y points for every x
        for ideck in range(len(midBeam)):  # For each deck in hull
            deckOutlinesHull.append([])  # For each deck create array
            if pdecks2[ideck] != 0:  # If not keel
                if transomBeam[ideck] > 0 and not use_forward_model_for_aft:  # Add vertical hull line at transom
                    deckOutlinesHull[ideck].append([aftEnd[ideck], 0])
                kmin = aftEnd[ideck]
                kmax = loa / 2
                if frame_positions is not None:
                    klist=[]
                    klist.append(kmin)
                    for pos in frame_positions:
                        if pos>kmin and pos<=kmax:
                            klist.append(pos)
                    #klist.append(kmax)
                else:
                    klist = np.linspace(kmin, kmax, nump)

                if use_forward_model_for_aft:
                    for xpt in klist:
                        eqX = (xpt - loa / 2) / ((loa / 2)-keel_stern+sternRake[ideck])
                        ypt = (1 - ((Math.cosh(eqX * sternDeckMArray[ideck] * Math.pi) - 1) / (
                                Math.cosh(sternDeckMArray[ideck] * Math.pi) - 1))) * midBeam[ideck]
                        deckOutlinesHull[ideck].append([xpt, ypt])
                else:
                    for xpt in klist:
                        ypt = Math.sqrt(
                            Math.pow(ogiveRadius[ideck], 2) -
                            Math.pow(xpt - loa / 2, 2)) + \
                              noseConeBaseRadius[ideck] - ogiveRadius[ideck] + \
                              transomBeam[ideck]
                        deckOutlinesHull[ideck].append([xpt, ypt])

                kmin = loa / 2
                kmax = keelFwd + bowRake[ideck]
                if frame_positions is not None:
                    klist=[]
                    #klist.append(kmin)
                    for pos in frame_positions:
                        if pos>kmin and pos<=kmax:
                            klist.append(pos)
                    klist.append(kmax)
                else:
                    klist = np.linspace(kmin, kmax, nump)
                for xpt in klist:
                    eqX = (xpt - loa / 2) / (keelFwd + bowRake[ideck] - (loa / 2))
                    ypt = (1 - ((Math.cosh(eqX * fwdDeckMArray[ideck] * Math.pi) - 1) / (
                                Math.cosh(fwdDeckMArray[ideck] * Math.pi) - 1)))* midBeam[ideck]
                    deckOutlinesHull[ideck].append([xpt,  ypt])


            else:  # If keel draw top
                kmin = aftEnd[ideck]
                kmax = (keelFwd + bowRake[ideck])
                if frame_positions is not None:
                 klist=[]
                 klist.append(kmin)
                 for pos in frame_positions:
                    if pos>kmin and pos<kmax:
                        klist.append(pos)
                 klist.append(kmax)
                else:
                    klist = np.linspace(kmin, kmax, nump*2)

                for xpt in klist:
                    deckOutlinesHull[ideck].append([xpt, 0])  # Straight line

        deckOutlinesS = []  # Array with superstructure deck outline x, y coordinates
        tumblehome = []  # Superstructure tumblehome
        for n in range(len(aftEndS)):  # For each deck in superstructure
            deckOutlinesS.append([])  # For each deck create array
            tumblehome = (pdecks3[n] - depth) * Math.tan(
                slope)  # Calculate tumblehome y offset to subtract below
            deckOutlinesS[n].append([aftEndS[n], 0])  # Add vertical hull line at transom

            kmin = aftEndS[n]
            kmax = loa / 2
            klist = np.linspace(kmin, kmax, nump)
            for xpt in klist:
                deckOutlinesS[n].append([xpt, (
                            Math.sqrt(Math.pow(ogiveRadius[0], 2) - Math.pow(xpt - loa / 2, 2)) +
                            noseConeBaseRadius[0] - ogiveRadius[0] + transomBeam[0] - tumblehome)])

            kmin = loa / 2
            kmax = (keelFwd + bowRakeS[n])
            klist = np.linspace(kmin, kmax, nump)
            for xpt in klist:
                eqX = (xpt - loa / 2) / (
                            keelFwd + bowRakeS[n] - (loa / 2))  # Value of x in JC equation
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
            #wlinesPos.append(wlineP)
            #wlinesNeg.append(wlineN)

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

    def readUCLDBBShipData(self):
        shipdata = {}
        pdecks = []
        dict_decks = {}
        pbulkheads = []
        with open(self.filename, newline='') as csvfile:
            f = csv.DictReader(csvfile)
            shipset = 0
            for row in f:  # there is only one row after header row!!!!!
                shipset = row

            shipdata["loa_val"] = float(shipset['LOA'])  # treba li učitavanje vrijednosti biti u petlji?
            shipdata["boa_val"] = float(shipset['BOA'])  # treba li učitavanje vrijednosti biti u petlji?
            shipdata['depth_val'] = float(shipset['D'])
            shipdata['shipname'] = shipset['Name']

            splitdata = str(shipset['HullData']).split(" ")
            shipdata["ms_val"] = float(splitdata[0])
            shipdata["bow_val"] = float(splitdata[1])
            shipdata["tr_stern_val"] = float(splitdata[2])
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
            #added
            shipdata['DeckPos'] = shipset['DeckPos']
            shipdata['BHPos'] = shipset['BHPos']
            shipdata['Decks'] = shipset['Decks']
            shipdata['Bulkheads'] = shipset['Bulkheads']
            shipdata['use_fwd_for_aft']=False
            shipdata['anc_keel_width'] = 0.0
            shipdata['anc_keel_height'] = 0.0
            shipdata['anc_rise_bow'] = 1.0
            shipdata['anc_rise_stern'] = 1.0
            if False:
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

        main_hull_depth = shipdata["depth_val"]
        splitdata = str(shipdata['DeckPos']).split(" ")
        nss = 0
        for dp in splitdata:
            z_deck = float(dp)
            pdecks.append(z_deck)
            if z_deck > main_hull_depth:
                nss += 1
        splitdata = str(shipdata['BHPos']).split(" ")
        for dp in splitdata:
            pbulkheads.append(float(dp))
        splitdata = str(shipdata['Decks']).split(" ")
        i = 0
        for z_deck in pdecks:
            id_deck = nss - i
            dict_decks[id_deck] = i
            i += 1

        return shipdata, pdecks, pbulkheads, dict_decks

    def read_file(self):
        shipdata = {}
        pdecks = []
        dict_decks = {}
        pbulkheads = []
        with open(self.filename, newline='') as csvfile:
            f = csv.DictReader(csvfile)
            shipset = 0
            for row in f:  # there is only one row after header row!!!!!
                shipset = row

            list_non_floats = {"DeckPos","BHPos","Decks","Bulkheads","shipname",'use_fwd_for_aft'}
            for key,value in shipset.items():
                if key in list_non_floats:
                    shipdata[key] = shipset[key]
                elif key =='use_fwd_for_aft':
                    shipdata[key] = bool(shipset[key])
                else:
                    shipdata[key] = float(shipset[key])

        depth = shipdata["depth_val"]
        splitdata = str(shipdata['DeckPos']).split(" ")
        nss = 0
        for dp in splitdata:
            z_deck = float(dp)
            pdecks.append(z_deck)
            if z_deck > depth:
                nss += 1
        splitdata = str(shipdata['BHPos']).split(" ")
        for dp in splitdata:
            pbulkheads.append(float(dp))
        splitdata = str(shipdata['Decks']).split(" ")
        i = 0
        for z_deck in pdecks:
            id_deck = nss - i
            dict_decks[id_deck] = i
            i += 1

        return shipdata, pdecks, pbulkheads, dict_decks


    def write_file(self, file_name):
        writecsv_dictionary(file_name,self.shipdata)

    def exportGeometry(self, fileName):
        filename_no_ext, file_extension = os.path.splitext(fileName)
        if file_extension == ".hgf":
            self.write_file(fileName)
        elif file_extension == ".stl" or file_extension == ".obj":
            HullFormFromMesh.export_hull_form(fileName,self)
