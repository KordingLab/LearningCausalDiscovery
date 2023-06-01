# Copyright (c) 2014 Greg James, Visual6502.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#cython: language_level=2, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, overflowcheck=False

import os, pickle, traceback
from array import array
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
#from libcpp.set cimport set as stdset

import cython
from cython.operator cimport dereference as deref


import copy
import sys



cdef extern from "cirsim.h" :
    cdef void test()

    cdef cppclass CPPGroup:
         CPPGroup() 
         vector[short] gvec
         int contains(int)
         void insert(int) 

cpdef enum WireState:
    PULLED_HIGH  = 1 << 0 # 1 
    PULLED_LOW     = 1 << 1 # 2
    GROUNDED       = 1 << 2 # 4
    HIGH           = 1 << 3 # 8
    FLOATING_HIGH  = 1 << 4 # 16
    FLOATING_LOW   = 1 << 5 # 32
    FLOATING       = 1 << 6 # 64

cdef int ANY_HIGH = (FLOATING_HIGH | HIGH | PULLED_HIGH)
cdef int ANY_LOW  = (FLOATING_LOW | GROUNDED | PULLED_LOW)
class Wire:

    def __init__(self, idIndex, name, controlTransIndices, transGateIndices, pulled):
        self.index = idIndex
        self.name = name

        # Transistors that switch other wires into connection with this wire
        self.ctInds = list(controlTransIndices)

        # Transistors whos gate is driven by this wire
        self.gateInds = list(transGateIndices) # FOR AN INSANE REASON WHEN THIS IS A SET IT FAILS

        # pulled reflects whether or not the wire is connected to
        # a pullup or pulldown.
        self._pulled = pulled

        # state reflects the logical state of the wire as the 
        # simulation progresses.
        self._state = pulled

    def __repr__(self):
        rstr = 'Wire %d "%s": %d  ct %s gates %s'%(self.index, self.name,
               self.state, str(self.ctInds), str(self.gateInds))
        return rstr


cdef int NMOS_GATE_LOW  = 0
cdef int NMOS_GATE_HIGH = 1 << 0

class NmosFet:
    GATE_LOW  = 0
    GATE_HIGH = 1 << 0

    def __init__(self, idIndex, side1WireIndex, side2WireIndex, gateWireIndex):
        
        # Wires switched together when this transistor is on
        self.side1WireIndex = side1WireIndex
        self.side2WireIndex = side2WireIndex
        self.gateWireIndex  = gateWireIndex

        self._gateState = NMOS_GATE_LOW
        self.index = idIndex

    def __repr__(self):
        rstr = 'NFET %d: %d gate %d [%d, %d]'%(self.index, self._gateState,
               self.gateWireIndex, self.side1WireIndex, self.side2WireIndex)
        return rstr



class CircuitSimulator(object):
    def __init__(self):
        self.name = ''
        self._wireList = None        # wireList[i] is a Wire.  wireList[i].index = i
        self._transistorList = None
        self.wireNames = dict()     # key is string wire names, value is integer wire index
        self.halfClkCount = 0       # the number of half clock cycles (low to high or high to low)
                                    # that the simulation has run
        self.tidx_lesion = -1 #int(input("Enter the lesion transistor id:"))

        # Performance / diagnostic info as sim progresses
        self.numAddWireToGroup = 0
        self.numAddWireTransistor = 0
        # General sense of how much work it's doing
        self.numWiresRecalculated = 0
        
        # If not None, call this to add a line to some log
        self.callback_addLogStr = None   # callback_addLogStr ('some text')

    def setLesionTransistor(self, lesion, tidx_lesion):
        if lesion is not None:
            if tidx_lesion != -1:
                self.lesion = lesion
                self.tidx_lesion = tidx_lesion
                print("Now the lesion transistor is: ",self.tidx_lesion)
            else:
                self.lesion = None
                print("Now there is no lesion transistor")
        else:
            self.lesion = None
            print("Now there is no lesion transistor")

    def setLesionTransistorForCirculator(self, lesion, tidx_lesion):
        self.calculator._setLesionTransistor(lesion, tidx_lesion)

    def setRecording(self, record_hr_state=False):
        # stop full resolution recording for this chip
        self.calculator._recordHRState = int(record_hr_state)

    def createStateArrays(self):
        # create the transistor and wire state arrays
        self._wireState = np.zeros(len(self._wireList), dtype=np.uint8)
        self._wireState[:] = [w._state for w in self._wireList]
        self._wirePulled = np.zeros(len(self._wireList), dtype=np.uint8)
        self._wirePulled[:] = [w._pulled for w in self._wireList] # fixme better way?

        self._transistorState = np.zeros(len(self._transistorList), dtype=np.uint8)
        self._transistorState[:] = [t._gateState for t in self._transistorList]

    def clearSimStats(self):
        self.numAddWireToGroup = 0
        self.numAddWireTransistor = 0

    def getWireIndex(self, wireNameStr):
        return self.wireNames[wireNameStr]

    def recalcNamedWire(self, wireNameStr):
        self.recalcWireList([self.wireNames[wireNameStr]])

    def recalcWireNameList(self, wireNameList):
        wireList = [None] * len(wireNameList)
        i = 0
        for name in wireNameList:
            wireList[i] = self.wireNames[name]
            i += 1
        self.recalcWireList (wireList)

    def recalcAllWires(self):
        """ Not fast.  Meant only for setting initial conditions """
        wireInds = []
        for ind, wire in enumerate(self._wireList):
            if wire is not None:
                wireInds.append(ind)
        self.recalcWireList (wireInds)
        
    def recalcWireList(self, nwireList):
        
        self.calculator.recalcWireList(nwireList, self.halfClkCount)

    def recalcWire(self, wireIndex):
        self.recalcWireList([wireIndex])

    def floatWire(self, wireIndex):
        i = wireIndex
        wire = self._wireList[i]

        if self._wirePulled[i] == PULLED_HIGH:
            self._wireState[i] = PULLED_HIGH
        elif self._wirePulled[i] == PULLED_LOW:
            self._wireState[i] = PULLED_LOW
        else:
            state = self._wireState[i]
            if state == GROUNDED or state == PULLED_LOW:
                self._wireState[i] = FLOATING_LOW
            if state == HIGH or state == PULLED_HIGH:
                self._wireState[i] = FLOATING_HIGH

    # setHighWN() and setLowWN() do not trigger an update
    # of the simulation.
    def setHighWN(self, n):
        if n in self.wireNames:
            wireIndex = self.wireNames[n]
            self._wireState[wireIndex] = PULLED_HIGH
            self._wirePulled[wireIndex] = PULLED_HIGH 


            return
        raise Exception("WHEN DO WE EVER GET HERE")

        # assert type(n) == type(1), 'wire thing %s'%str(n)
        # wire = self._wireList[n]
        # if wire is not None:
        #     wire._setHigh()
        # else:
        #     print 'ERROR - trying to set wire None high'

    def setLowWN(self, n):
        #FIXME WHAT THE HELL IS THIS ? 
        if n in self.wireNames:
            wireIndex = self.wireNames[n]
            self._wireState[wireIndex] = PULLED_LOW
            self._wirePulled[wireIndex] = PULLED_LOW 

            return
        raise Exception("WHEN DO WE EVER GET HERE")
        # assert type(n) == type(1), 'wire thing %s'%str(n)
        # wire = self._wireList[n]
        # if wire is not None:
        #     wire.setLow()
        # else:
        #     print 'ERROR - trying to set wire None low'

    def _setPulledHighOrLow(self, idx, boolHigh):
        if boolHigh == True:
            self._wirePulled[idx] = PULLED_HIGH
            self._wireState[idx]  = PULLED_HIGH
        elif boolHigh == False:
            self._wirePulled[idx] = PULLED_LOW
            self._wireState[idx]  = PULLED_LOW

    def setHigh(self, wireIndex):
        self._setPulledHighOrLow(wireIndex, True)

    def setLow(self, wireIndex):
        self._setPulledHighOrLow(wireIndex, False)

    def setPulled(self, wireIndex, boolHighOrLow):
        self._setPulledHighOrLow(wireIndex, boolHighOrLow)

    def setPulledHigh(self, wireIndex):
        self._setPulledHighOrLow(wireIndex, True)

    def setPulledLow(self, wireIndex):
        self._setPulledHighOrLow(wireIndex, False)

    def isHigh(self, wireIndex):
        return bool((self._wireState[wireIndex] & (ANY_HIGH)))

    def isLow(self, wireIndex):
        return bool( self._wireState[wireIndex] & ANY_LOW)

    def isHighWN(self, n):
        raise NotImplementedError()
        if n in self.wireNames:
            wireIndex = self.wireNames[n]
            return self._wireList[wireIndex]._isHigh()

        assert type(n) == type(1), 'ERROR: if arg to isHigh is not in ' + \
            'wireNames, it had better be an integer'
        wire = self._wireList[n]
        assert wire is not None
        return wire.isHigh()
        
    def isLowWN(self, n):
        raise NotImplementedError()
        if n in self.wireNames:
            wireIndex = self.wireNames[n]
            return self._wireList[wireIndex]._isLow()

        wire = self._wireList[n]
        assert wire is not None
        return wire.isLow()

    # TODO: rename to getNamedSignal (name, lowBitNum, highBitNum) ('DB',0,7) 
    # TODO: elim or use wire indices
    # Use for debug and to examine busses.  This is slow. 
    def getGen(self, strSigName, size):
        raise NotImplementedError()
        data = 0
        for i in xrange(size, -1, -1):
            data = data * 2
            bit = '%s%d'%(strSigName,i)
            if self.isHighWN(bit):
                data = data + 1
        return data

    def setGen(self, data, string, size):
        raise NotImplementedError()
        d = data
        for i in xrange(size):
            bit = '%s%d'%(string,i)
            if (d & 1) == 1:
                self.setHigh(bit)
            else:
                self.setLowWN(bit)
            d = d / 2
            
    def updateWireNames (self, wireNames):        
        for j in wireNames:
            i = 0
            nameStr = j[0]
            for k in j[1:]:
                name = '%s%d'%(nameStr,i)
                self._wireList[k].name = name
                self.wireNames[name] = k
                i += 1

    def getWiresState(self):
        return np.array(self._wireState)
 
    def getPulledState(self):
        return np.array(self._wirePulled)

    def getTransistorState(self):
        return np.array(self._transistorState)

    def getFullTransistorState(self):
        # convert cython memoryview to numpy array
        array = np.array(self.calculator.getFullTransistorState())
        return array

    def resetFullTransistorState(self):
        self.calculator.resetFullTransistorState()

    def intervenTransistor(self, transistorIndex, state):
        self.calculator.intervenTransistor(transistorIndex, state)

    def loadCircuit (self, filePath):

        if not os.path.exists(filePath):
            raise Exception('Could not find circuit file: %s  from cwd %s'%
                            (filePath, os.getcwd()))
        print 'Loading %s' % filePath
        
        of = open (filePath, 'rb')
        rootObj = pickle.load (of)
        of.close()

        numWires = rootObj['NUM_WIRES']
        nextCtrl = rootObj['NEXT_CTRL']
        noWire = rootObj['NO_WIRE']
        wirePulled = rootObj['WIRE_PULLED']
        wireCtrlFets = rootObj['WIRE_CTRL_FETS']
        wireGates = rootObj['WIRE_GATES']
        wireNames = rootObj['WIRE_NAMES']
        numFets = rootObj['NUM_FETS']
        fetSide1WireInds = rootObj['FET_SIDE1_WIRE_INDS']
        fetSide2WireInds = rootObj['FET_SIDE2_WIRE_INDS']
        fetGateWireInds = rootObj['FET_GATE_INDS']
        numNoneWires = rootObj['NUM_NULL_WIRES']

        l = len(wirePulled)
        assert l == numWires, 'Expected %d entries in wirePulled, got %d'%(numWires, l)
        l = len(wireNames)
        assert l == numWires, 'Expected %d wireNames, got %d'%(numWires, l)

        l = len(fetSide1WireInds)
        assert l == numFets, 'Expected %d fetSide1WireInds, got %d'%(numFets, l)
        l = len(fetSide2WireInds)
        assert l == numFets, 'Expected %d fetSide2WireInds, got %d'%(numFets, l)
        l = len(fetGateWireInds)
        assert l == numFets, 'Expected %d fetGateWireInds, got %d'%(numFets, l)

        self._wireList = [None] * numWires

        i = 0
        wcfi = 0
        wgi = 0
        while i < numWires:
            numControlFets = wireCtrlFets[wcfi]
            wcfi += 1
            controlFets = set()
            n = 0
            while n < numControlFets:
                controlFets.add(wireCtrlFets[wcfi])
                wcfi += 1
                n += 1
            tok = wireCtrlFets[wcfi]
            wcfi += 1
            assert tok == nextCtrl, 'Wire %d read 0x%X instead of 0x%X at end of ctrl fet segment len %d: %s'%(
                i, tok, nextCtrl, numControlFets, str(wireCtrlFets[wcfi-1-numControlFets-1:wcfi]))

            numGates = wireGates[wgi]
            wgi += 1
            gates = set()
            n = 0
            while n < numGates:
                gates.add(wireGates[wgi])
                wgi += 1
                n += 1
            tok = wireGates[wgi]
            wgi += 1
            assert tok == nextCtrl, 'Wire %d Read 0x%X instead of 0x%X at end of gates segment len %d: %s'%(
                i, tok, nextCtrl, numGates, str(wireGates[wgi-1-numGates-1:wgi]))

            if len(wireCtrlFets) == 0 and len(gates) == 0:
                assert wireNames[i] == ''
                self._wireList[i] = None
            else:
                self._wireList[i] = Wire(i, wireNames[i], controlFets, gates, wirePulled[i])
                self.wireNames[wireNames[i]] = i
            i += 1

        self._transistorList = [None] * numFets
        i = 0
        while i < numFets:
            s1 = fetSide1WireInds[i]
            s2 = fetSide2WireInds[i]
            gate = fetGateWireInds[i]
            if s1 == noWire:
                assert s2 == noWire
                assert gate == noWire
            else:
                # if i == self.tidx_lesion:
                #     if self.lesion == 'High':
                #         # Connect the lesion gate to the VCC, force it to be HIGH
                #         self._transistorList[i] = NmosFet(i, s1, s2, self.wireNames['VCC'])
                #         self._wireList[gate].gateInds.remove(i)
                #         self._wireList[self.wireNames['VCC']].gateInds.append(long(i))
                #     elif self.lesion == 'Low':
                #         # Connect the lesion gate to the GND, force it to be LOW
                #         self._transistorList[i] = NmosFet(i, s1, s2, self.wireNames['VSS'])
                #         self._wireList[gate].gateInds.remove(i)
                #         self._wireList[self.wireNames['VSS']].gateInds.append(long(i))
                #     else:
                #         raise Exception('Unknown lesion type: %s'%self.lesion)
                # else:
                #     self._transistorList[i] = NmosFet(i, s1, s2, gate)
                self._transistorList[i] = NmosFet(i, s1, s2, gate)
            i += 1

        assert 'VCC' in self.wireNames
        assert 'VSS' in self.wireNames
        self.vccWireIndex = self.wireNames['VCC']
        self.gndWireIndex = self.wireNames['VSS']
        self._wireList[self.vccWireIndex]._state = HIGH
        self._wireList[self.gndWireIndex]._state = GROUNDED

        for transInd in self._wireList[self.vccWireIndex].gateInds:
            self._transistorList[transInd]._gateState = NMOS_GATE_HIGH

        # initialize the gate state to be lesion
        if self.lesion == 'High':
            self._transistorList[self.tidx_lesion]._gateState = NMOS_GATE_HIGH
        elif self.lesion == 'Low':
            self._transistorList[self.tidx_lesion]._gateState = NMOS_GATE_LOW

        self.lastWireGroupState = [-1] * numWires

        # create the calculator
        self.createStateArrays()
        self.calculator  = WireCalculator(self._wireList,
                                          self._transistorList,
                                          self._wireState,
                                          self._wirePulled,
                                          self._transistorState,
                                          self.gndWireIndex,
                                          self.vccWireIndex)

        return rootObj


    def writeCktFile(self, filePath):
 
        rootObj = dict()
        
        numWires = len(self._wireList)
        nextCtrl = 0xFFFE

        # 'B' for unsigned integer, minimum of 1 byte
        wirePulled = array('B', [0] * numWires)

        # 'I' for unsigned int, minimum of 2 bytes
        wireControlFets = array('I')
        wireGates = array('I')
        numNoneWires = 0
        wireNames = []

        for i, wire in enumerate(self._wireList):
            if wire is None:
                wireControlFets.append(0)
                wireControlFets.append(nextCtrl)
                wireGates.append(0)
                wireGates.append(nextCtrl)
                numNoneWires += 1
                wireNames.append('')
                continue

            wirePulled[i] = wire.pulled

            wireControlFets.append(len(wire.ins))
            for transInd in wire.ins:
                wireControlFets.append(transInd)
            wireControlFets.append(nextCtrl)

            wireGates.append(len(wire.outs))
            for transInd in wire.outs:
                wireGates.append(transInd)
            wireGates.append(nextCtrl)

            wireNames.append(wire.name)

        noWire = 0xFFFD
        numFets = len(self._transistorList)
        fetSide1WireInds = array('I', [noWire] * numFets)
        fetSide2WireInds = array('I', [noWire] * numFets)
        fetGateWireInds  = array('I', [noWire] * numFets)

        for i, trans in enumerate(self._transistorList):
            if trans is None:
                continue
            fetSide1WireInds[i] = trans.c1
            fetSide2WireInds[i] = trans.c2
            fetGateWireInds[i] = trans.gate

        rootObj['NUM_WIRES'] = numWires
        rootObj['NEXT_CTRL'] = nextCtrl
        rootObj['NO_WIRE'] = noWire
        rootObj['WIRE_PULLED'] = wirePulled
        rootObj['WIRE_CTRL_FETS'] = wireControlFets
        rootObj['WIRE_GATES'] = wireGates
        rootObj['WIRE_NAMES'] = wireNames
        rootObj['NUM_FETS'] = numFets
        rootObj['FET_SIDE1_WIRE_INDS'] = fetSide1WireInds
        rootObj['FET_SIDE2_WIRE_INDS'] = fetSide2WireInds
        # Extra info to verify the data and connections
        rootObj['FET_GATE_INDS'] = fetGateWireInds
        rootObj['NUM_NULL_WIRES'] = numNoneWires

        of = open(filePath, 'wb')
        pickle.dump(rootObj, of)
        of.close()


# cdef inline int group_contains(stdset[int] & group, int x):
#     cdef int i
#     return group.find(x) != group.end()
    
#     for i in group:
#          if x == i:
#              return True
#     return False
         
# cdef inline void group_insert(stdset[int] & group, int x):
#     group.insert(x)
cdef struct Group:
    vector[int] gvec


cdef inline int group_contains(Group & group, int x):
    cdef int i
    for i in group.gvec:
         if x == i:
             return True
    return False
         
cdef inline void group_insert(Group & group, int x):
    cdef int i
    for i in group.gvec:
         if x == i:
             return 
    group.gvec.push_back(x)


#cpdef enum TransistorIndexPos:
cdef int    TW_GATE = 0
cdef int    TW_S1 = 1
cdef int    TW_S2 = 2

@cython.final
cdef class WireCalculator:
    cdef object _wireList
    cdef np.uint8_t[:] _wireState

    cdef np.uint8_t[:] _wirePulled
    cdef np.uint8_t[:] _transistorState
    cdef np.uint8_t[:] recalcArray
    cdef int gndWireIndex
    cdef int vccWireIndex
    cdef int numAddWireToGroup
    cdef int numAddWireTransistor
    cdef int numWiresRecalculated
    cdef object callback_addLogStr
    cdef int recalcCap
    cdef vector[int] recalcOrderStack
    cdef np.uint8_t[:] newRecalcArray
    cdef vector[int] newRecalcOrderStack
    cdef np.int32_t[:, :] _transistorWires
    cdef np.int32_t[:] _numWires
    # this is a hack because WHO knows how to do this ? NOT ME
    cdef np.int32_t[:, :] _ctInds
    cdef np.int32_t[:, :] _gateInds
    cdef int _latestHalfClkCount
    # # inner state within a half clock
    cdef public int _tidx_lesion
    cdef public int _lesion_type
    cdef public int _recordHRState
    cdef np.uint8_t[:, :] _full_transistor_state
    cdef unsigned long _n_iters

    def __init__(self, wireList, transistorList, 
                 wireState, wirePulled, transistorState, # all references
                 gndWireIndex,
                 vccWireIndex,):

        self._numWires = np.zeros(len(wireList), dtype=np.int32)

        self._wireList = wireList
        #self._transistorList = transistorList
        self._wireState = wireState

        self._wirePulled = wirePulled
        self._transistorState = transistorState

        self.recalcArray = None

        self.gndWireIndex = gndWireIndex
        self.vccWireIndex = vccWireIndex

        # lesion index

        # Performance / diagnostic info as sim progresses
        self.numAddWireToGroup = 0
        self.numAddWireTransistor = 0
        # General sense of how much work it's doing
        self.numWiresRecalculated = 0
        
        # If not None, call this to add a line to some log
        self.callback_addLogStr = None   # callback_addLogStr ('some text')

        self.recalcCap = len(self._transistorState)
        # Using lists [] for these is faster than using array('B'/'L', ...)
        self.recalcArray = np.zeros(self.recalcCap, dtype=np.uint8) # [False] * self.recalcCap
        self.recalcOrderStack.reserve(4000)
        self.newRecalcArray = np.zeros(self.recalcCap, dtype=np.uint8) # [0] * self.recalcCap
        self.newRecalcOrderStack.reserve(4000)

        self._ctInds = np.zeros((len(wireList), 4000), dtype=np.int32)
        self._gateInds = np.zeros((len(wireList), 4000), dtype=np.int32)

        # inner state within a half clock
        self._tidx_lesion = -1
        self._lesion_type = 0
        self._recordHRState = 0
        self._full_transistor_state = np.zeros((len(transistorState), 50000), dtype=np.uint8)
        self._n_iters = 0

        # count the wires
        for wi, w in enumerate(wireList):
            self._numWires[wi] = len(w.ctInds) + len(w.gateInds)

            self._ctInds[wi, 0] = len(w.ctInds)
            for i, cti in enumerate(w.ctInds):
                self._ctInds[wi, i+1] = cti

            self._gateInds[wi, 0] = len(w.gateInds)
            for i, gi in enumerate(w.gateInds):
                self._gateInds[wi, i+1] = gi
        self._prepForRecalc()


        # create the transistor index array
        self._transistorWires = np.zeros((len(transistorList), 
                                          3), dtype=np.int32)
        for ti, t in enumerate(transistorList):
            self._transistorWires[ti, TW_GATE] = t.gateWireIndex
            self._transistorWires[ti, TW_S1] = t.side1WireIndex
            self._transistorWires[ti, TW_S2] = t.side2WireIndex
            

        self._latestHalfClkCount = 0


    cdef _prepForRecalc(self):
        self.recalcOrderStack.clear() #  = []
        self.newRecalcOrderStack.clear() #  = []
        

    def recalcWireList(self, nwireList, halfClkCount):

        self._prepForRecalc()

        for wireIndex in nwireList:
            # recalcOrder is a list of wire indices.  self.lastRecalcOrder
            # marks the last index into this list that we should recalculate.
            # recalcArray has entries for all wires and is used to mark
            # which wires need to be recalcualted.
            self.recalcOrderStack.push_back(wireIndex)
            self.recalcArray[wireIndex] = True
        self._doRecalcIterations(halfClkCount)


    cdef void _doRecalcIterations(self, int halfClkCount):
        # Simulation is not allowed to try more than 'stepLimit' 
        # iterations.  If it doesn't converge by then, raise an 
        # exception.
        cdef int step = 0
        cdef int stepLimit = 50
        cdef int a, b
        cdef int i, s



        self._latestHalfClkCount = halfClkCount
        while step < stepLimit:
            # print 'Iter %d, num to recalc %d ' %(step, len(self.recalcOrderStack))

            if self.recalcOrderStack.empty():
                break;

            for wireIndex in self.recalcOrderStack:
                # print('*'*50)

                self.newRecalcArray[wireIndex] = 0
                self._doWireRecalc(wireIndex)

                self.recalcArray[wireIndex] = False
                self.numWiresRecalculated += 1

            # experimental highest resolution recording of transistor states
            if self._recordHRState:
                self._recordTransistorState()

            for i in range(self.recalcCap):
                a = self.recalcArray[i]
                b = self.newRecalcArray[i]
                self.newRecalcArray[i] = a
                self.recalcArray[i] = b

            self.recalcOrderStack = self.newRecalcOrderStack
            self.newRecalcOrderStack.clear()

            step += 1

        # The first attempt to compute the state of a chip's circuit
        # may not converge, but it's enough to settle the chip into
        # a reasonable state so that when input and clock pulses are
        # applied, the simulation will converge.

        # (for causal analysis) not converge means the state of the
        # chip has been changed, since the regular state converges
        # so catch the error and return full length state
        if step >= stepLimit:
            msg = 'ERROR: Sim  did not converge after %d iterations'% \
                  ( stepLimit)
            if self.callback_addLogStr:
                self.callback_addLogStr(msg)
            # Don't raise an exception if this is the first attempt
            # to compute the state of a chip, but raise an exception if
            # the simulation doesn't converge any time other than that.
            if halfClkCount > 0:
                traceback.print_stack()
                raise RuntimeError(msg)

        # Check that we've properly reset the recalcArray.  All entries
        # should be zero in preparation for the next half clock cycle.
        # FIXME WE SHOULD REALLY DO THIS
        # Only do this sanity check for the first clock cycles.
        if halfClkCount < 20:
            needNewArray = False
            for recalc in self.recalcArray:
                if recalc != False:
                    needNewArray = True
                    if step < stepLimit:
                        msg = 'ERROR: at halfclk %d, '%(halfClkCount) + \
                              'after %d iterations'%(step) + \
                              'an entry in recalcArray is not False at the ' + \
                              'end of an update'
                        print(msg)
                        break
            if needNewArray:
                print "OMG WE NEEDED A NEW ARRAY"
                self.recalcArray = np.zeros_like(self.recalcArray) # [False] * len(self.recalcArray)
                



    cdef void _floatWire(self, int wireIndex):
        cdef int i = wireIndex
        cdef int state = self._wireState[i]

        if self._wirePulled[i] == PULLED_HIGH:
            self._wireState[i] = PULLED_HIGH
        elif self._wirePulled[i] == PULLED_LOW:
            self._wireState[i] = PULLED_LOW
        else:
            if state == GROUNDED or state == PULLED_LOW:
                self._wireState[i] = FLOATING_LOW
            if state == HIGH or state == PULLED_HIGH:
                self._wireState[i] = FLOATING_HIGH


    cdef void _doWireRecalc(self, int wireIndex):
        cdef int transIndex
        cdef int newValue, newHigh, gateState
        cdef int groupWireIndex, gate_inds_cnt, ti
        #print "_doWireRecalc(%d)" % wireIndex
        if wireIndex == self.gndWireIndex or wireIndex == self.vccWireIndex:
            return
        
        cdef CPPGroup group #  = set()

        # addWireToGroup recursively adds this wire and all wires
        # of connected transistors
        self._addWireToGroup(wireIndex, group)
        
        newValue = self._getWireValue(group)
        newHigh = newValue == HIGH or newValue == PULLED_HIGH or \
                  newValue == FLOATING_HIGH

        for groupWireIndex in group.gvec:
            if groupWireIndex == self.gndWireIndex or \
               groupWireIndex == self.vccWireIndex:
                # TODO: remove gnd and vcc from group?
                continue
            #simWire = self._wireList[groupWireIndex]
            #simWire.state = newValue
            self._wireState[groupWireIndex] = newValue
            gate_inds_cnt = self._gateInds[groupWireIndex, 0]
            # print('*'*80)
            for ti in range(gate_inds_cnt): # in simWire.gateInds:
                transIndex = self._gateInds[groupWireIndex, ti+1]
                gateState = self._transistorState[transIndex]

                if newHigh == True and gateState == NMOS_GATE_LOW:
                    self._turnTransistorOn(transIndex)
                if newHigh == False and gateState == NMOS_GATE_HIGH:
                    self._turnTransistorOff(transIndex)

    cdef void _scheduleWireRecalc(self, int wireInd):
        """
        push this wire onto the recalc order stack if its not
        already there
        """
    
        if self.newRecalcArray[wireInd] == 0:
            self.newRecalcArray[wireInd] = 1
            self.newRecalcOrderStack.push_back(wireInd)
    
    cdef void _turnTransistorOn(self, int tidx):
        cdef int wireInd
        # print "_turnTransistorOn(%d)" % tidx
        if tidx == self._tidx_lesion:
            if self._lesion_type == 2:
                self._transistorState[tidx] = NMOS_GATE_HIGH
            elif self._lesion_type == 1:
                self._transistorState[tidx] = NMOS_GATE_LOW
            elif self._lesion_type == 0:
                pass
            else:
                raise Exception("Unknown lesion type: %s" % self._lesion_type)
        else:
            self._transistorState[tidx] = NMOS_GATE_HIGH

        cdef int c1Wire = self._transistorWires[tidx, TW_S1]
        cdef int c2Wire = self._transistorWires[tidx, TW_S2]

        self._scheduleWireRecalc(c1Wire)

        self._scheduleWireRecalc(c2Wire)

    cdef void _turnTransistorOff(self, int tidx):
        # print "_turnTransistorOff(%d)" % tidx
        if tidx == self._tidx_lesion:
            if self._lesion_type == 2:
                self._transistorState[tidx] = NMOS_GATE_HIGH
            elif self._lesion_type == 1:
                self._transistorState[tidx] = NMOS_GATE_LOW
            elif self._lesion_type == 0:
                pass
            else:
                raise Exception("Unknown lesion type: %s" % self._lesion_type)
        else:
            self._transistorState[tidx] = NMOS_GATE_LOW

        cdef int wireInd

        #t = self._transistorList[tidx]
        cdef int c1Wire = self._transistorWires[tidx, TW_S1]
        cdef int c2Wire = self._transistorWires[tidx, TW_S2]

        self._floatWire(c1Wire)
        self._floatWire(c2Wire)

        self._scheduleWireRecalc(c1Wire)
        self._scheduleWireRecalc(c2Wire)

    cdef int _getWireValue(self, CPPGroup & group):
        """
        This function performs group resolution for a collection
        of wires

        if any wire in the group is ground, return grounded
        if any wire in the group is vcc and it's not grounded, 
        return high; else return grounded

        
        """
        # TODO PERF: why turn into a list?
        #l = list(set(group))
        #print "getting wire value of group of size", group.size()
        cdef int sawFl = False
        cdef int sawFh = False
        cdef int firstval = deref(group.gvec.begin())
        cdef int value = self._wireState[firstval] # l[0]]
        cdef int wire_pulled, wire_state
        for wireIndex in group.gvec:
            if wireIndex == self.gndWireIndex:
                return GROUNDED
            if wireIndex == self.vccWireIndex:
                if group.contains(self.gndWireIndex):
                    return GROUNDED
                else:
                    return HIGH
                
            wire_pulled = self._wirePulled[wireIndex]
            wire_state = self._wireState[wireIndex]
            if wire_pulled == PULLED_HIGH:
                value = PULLED_HIGH
            elif wire_pulled == PULLED_LOW:
                value = PULLED_LOW
                
            if wire_state == FLOATING_LOW:
                sawFl = True
            elif wire_state == FLOATING_HIGH:
                sawFh = True

        if value == FLOATING_LOW or value == FLOATING_HIGH:
            # If two floating regions are connected together,
            # set their voltage based on whichever region has
            # the most components.  The resulting voltage should
            # be determined by the capacitance of each region.
            # Instead, we use the count of the number of components
            # in each region as an estimate of how much charge 
            # each one holds, and set the result hi or low based
            # on which region has the most components.
            if sawFl and sawFh:
                value = self._countWireSizes(group)
                # if sizes[1] < sizes[0]:
                #     value = FLOATING_LOW
                # else:
                #     value = FLOATING_HIGH
        return value


    cdef void _addWireToGroup(self, int wireIndex, CPPGroup & group):
        # Add this wire to the group. 
        cdef int ctind, ctIndsSize, t
        #print "addWireToGroup(%d)" % wireIndex
        self.numAddWireToGroup += 1
        group.insert(wireIndex)

        if wireIndex == self.gndWireIndex or wireIndex == self.vccWireIndex:
            return

        # for each transistor which switch other wires into connection
        # with this wire -- if the tarnsistor is on, add in the switched
        # wire to the group
        ctIndsSize =self._ctInds[wireIndex, 0]
        for t in range(ctIndsSize):
            ctind = self._ctInds[wireIndex, t+1]
            self._addWireTransistor (wireIndex, ctind, group)

    cdef void _addWireTransistor(self, int wireIndex, int t, CPPGroup & group):
        # for this wire and this transistor, check if the transistor
        # is on. If it is, it has then made a connection between the
        #  wires connected to C1 and C2. 

        # This appears to only be triggered for C1/C2 wires
        #print "addWireTransistor(wireIndex=%d, t=%d)" % (wireIndex, t)
        self.numAddWireTransistor += 1
        cdef int other = -1
        #trans = self._transistorList[t]
        cdef int c1Wire = self._transistorWires[t, TW_S1]
        cdef int c2Wire = self._transistorWires[t, TW_S2]

        if self._transistorState[t] == NMOS_GATE_LOW:
            return
        if c1Wire == wireIndex:
            other = c2Wire
        if c2Wire == wireIndex:
            other = c1Wire
        if other == self.vccWireIndex or other == self.gndWireIndex:
            group.insert(other)
            return
        if group.contains(other): 
            return
        self._addWireToGroup(other, group)

    cdef int _countWireSizes(self, CPPGroup & group):
        ##print "_countWireSizes group.size()=", group.size()
        cdef int countFl = 0
        cdef int countFh = 0
        cdef int i = 0
        cdef int num = 0
        cdef int wire_state = 0
        for i in group.gvec:
            wire_state = self._wireState[i]
            num = self._numWires[i]
            if wire_state == FLOATING_LOW:
                countFl += num
            if wire_state == FLOATING_HIGH:
                countFh += num
        if countFh < countFl:
            return FLOATING_LOW
        else:
            return FLOATING_HIGH

    cdef void _recordTransistorState(self):
        # Record the state of all transistors in the circuit.
        self._full_transistor_state[:, self._n_iters] = self._transistorState[:]
        self._n_iters = self._n_iters + 1

    def getFullTransistorState(self):
        # Return the full state of the transistors and clear the state storage
        self._full_transistor_state[:, self._n_iters] = -1
        res = self._full_transistor_state[:, :self._n_iters+1]
        self._full_transistor_state = np.zeros_like(self._full_transistor_state)
        self._n_iters = 0
        return res

    def resetFullTransistorState(self):
        self._full_transistor_state = np.zeros_like(self._full_transistor_state)
        self._n_iters = 0

    def _setLesionTransistor(self, lesion, tidx_lesion):
        lesionstr2int = {'None': 0, 'Low': 1, 'High': 2}
        if lesion is not None:
            self._lesion_type = lesionstr2int[str(lesion)] # 0 - No lesion, 1 - Low lesion, 2 - High lesion
            self._tidx_lesion = tidx_lesion


    def intervenTransistor(self, int transistor_idx, int state):
        if state == 1:  # Turn the transistor on
            # self._turnTransistorOn(transistor_idx)
            self._transistorState[transistor_idx] = NMOS_GATE_HIGH
        elif state == 0:  # Turn the transistor off
            # self._turnTransistorOff(transistor_idx)
            self._transistorState[transistor_idx] = NMOS_GATE_LOW

        self._lesion_type = state + 1
        self._tidx_lesion = transistor_idx

        c1Wire = self._transistorWires[transistor_idx, TW_S1]
        c2Wire = self._transistorWires[transistor_idx, TW_S2]
        self._doWireRecalc(c1Wire)
        self._doWireRecalc(c2Wire)