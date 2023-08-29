import numpy
import scipy
import h5py

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path

import SOAPify.HDF5er as HDF5er
from SOAPify import saponifyTrajectory



def transposeAndFlatten(x, /):
    trans = numpy.transpose(x)
    trans_fl = numpy.reshape(trans, numpy.shape(trans)[0] * numpy.shape(trans)[1])
    return trans_fl

def prepareData(x, /):
    """prepares an array from shape (atom,frames) to  (frames,atom)"""
    shape = x.shape
    toret = numpy.empty((shape[1], shape[0]), dtype=x.dtype)
    for i, atomvalues in enumerate(x):
        toret[:, i] = atomvalues
    return toret


# classyfing by knoing the min/max of the clusters
def classifying(x, classDict):
    toret = numpy.ones_like(x, dtype=int) * (len(classDict) - 1)
    # todo: sort  by max and then classify
    minmax = [[cname, data["max"]] for cname, data in classDict.items()]
    minmax = sorted(minmax, key=lambda x: -x[1])
    # print(minmax)
    for cname, myMax in minmax:
        toret[x < myMax] = int(cname)
    return toret

def signaltonoise(a: numpy.array, axis, ddof):
    """Given an array, retunrs its signal to noise value of its compontens"""
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * numpy.log10(abs(numpy.where(sd == 0, 0, m / sd)))

def export(outfile, trajFileName, wantedTrajectory, GROUP, XYZ_ORIG, classifiedFilteredLENS):
    # as a function, so the ref universe should be garbage collected correctly
    with h5py.File(trajFileName, "r") as trajFile, open(
        outfile, "w") as xyzFile:
        from MDAnalysis.transformations import fit_rot_trans

        tgroup = trajFile[GROUP]
        ref = HDF5er.createUniverseFromSlice(tgroup, [0])
        nAt= len(ref.atoms)
        ref.add_TopologyAttr("mass", [1] * nAt)
        # load antohter univer to avoid conatmination in the main one
        exportuniverse = HDF5er.createUniverseFromSlice(tgroup, slice(0, None, None))
        exportuniverse.add_TopologyAttr("mass", [1] * nAt)
        exportuniverse.trajectory.add_transformations(fit_rot_trans(exportuniverse, ref))
        HDF5er.getXYZfromMDA(
            xyzFile,
            exportuniverse,
            framesToExport=wantedTrajectory,
            allFramesProperty=XYZ_ORIG,
            proposedClassification=prepareData(classifiedFilteredLENS),
        )
        #universe.trajectory

def getDensity(data, PCX, PCY, bins, sigma=1):
    h, xe, ye = numpy.histogram2d(data[:, PCX], data[:, PCY], bins=bins, density=True)
    lh = numpy.log(h)
    lhmax = numpy.max((lh))
    lh -= lhmax
    lhmin = numpy.min((lh[lh != -numpy.inf]))
    lh[lh == -numpy.inf] = lhmin
    scipy.ndimage.gaussian_filter(lh, sigma=sigma, order=0, output=lh)
    X = getRanges(xe)
    Y = getRanges(ye)
    return lh.T, X, Y, lhmin

def getRanges(interval):
    return (interval[:-1] + interval[1:]) / 2


def _orderByWeight(data_matrix: numpy.ndarray) -> numpy.ndarray:
    toret = numpy.zeros_like(data_matrix, dtype=int)
    for i in range(toret.shape[0]):
        toret[i] = numpy.argsort(data_matrix[i])

    return toret[:, ::-1]


def _orderByWeightReverse(data_matrix: numpy.ndarray) -> numpy.ndarray:
    toret = numpy.zeros_like(data_matrix, dtype=int)
    for i in range(toret.shape[0]):
        toret[i] = numpy.argsort(data_matrix[i])

    return toret


def _orderByNone(data_matrix: numpy.ndarray) -> numpy.ndarray:
    toret = numpy.zeros_like(data_matrix, dtype=int)
    for i in range(toret.shape[0]):
        toret[:, i] = i

    return toret


def _orderByPosition(data_matrix: numpy.ndarray) -> numpy.ndarray:
    n = data_matrix.shape[0]
    t = list(range(n))
    half = n // 2
    toret = numpy.zeros_like(data_matrix, dtype=int)
    for i in range(toret.shape[0]):
        f = -(half - i)
        toret[:, i] = numpy.array(t[f:] + t[:f])
    return toret  # [:, ::-1]


def _prepareBases(
    data_matrix: numpy.ndarray, gap: float
) -> "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]":
    """get the ideogram ends in degrees

    Args:
        data_matrix (numpy.ndarray): the working matrix
        gap (float): the gap between the id, in degrees

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: returns the sum of the row, the length of the bases in degrees and the angular limits of the bases
    """
    L = data_matrix.shape[1]
    row_sum = numpy.sum(data_matrix, axis=1)
    ideogram_length = (360.0 - gap * L) * row_sum / numpy.sum(row_sum)
    ideo_ends = numpy.zeros((len(ideogram_length), 2))
    left = 0
    for k in range(len(ideogram_length)):
        right = left + ideogram_length[k]
        ideo_ends[k, :] = [left, right]
        left = right + gap
    return row_sum, ideogram_length, ideo_ends


def _bezierArcMaker(
    start: "numpy.array", end: "numpy.array", center: "numpy.array"
) -> "list[numpy.array]":
    """gets the two mid control points for generating an approximation of an arc with a cubic bezier

    Args:
        start (numpy.array): the start point
        end (numpy.array): the end point
        center (numpy.array): the center of the circumference

    Returns:
        list[numpy.array]: _description_
    """
    # source https://stackoverflow.com/a/44829356

    # TODO: se up a maximum 45 or 90 degree for the aproximation
    c = numpy.array([center[0], center[1]], dtype=float)
    p1 = numpy.array([start[0], start[1]], dtype=float)
    p4 = numpy.array([end[0], end[1]], dtype=float)
    a = p1 - c
    b = p4 - c
    q1 = a.dot(a)
    q2 = q1 + a.dot(b)
    k2 = (4.0 / 3.0) * (numpy.sqrt(2 * q1 * q2) - q2) / (a[0] * b[1] - a[1] * b[0])
    p2 = c + a + k2 * numpy.array([-a[1], a[0]])
    p3 = c + b + k2 * numpy.array([b[1], -b[0]])
    return [p2, p3]


def _ribbonCoordMaker(
    data_matrix: numpy.ndarray,
    row_value: numpy.ndarray,
    ideogram_length: numpy.ndarray,
    ideo_ends: numpy.ndarray,
    ignoreLessThan: int = 1,
    onlyFlux: bool = False,
    ordering: str = "matrix",
) -> "list[dict]":
    # TODO:add ordering:
    # - matrix do not does nothing
    # - leftright put the self in the center
    # - weight reorders the entries for each from the bigger
    # - weightr reorders the entries for each from the smaller
    orders = _orderByNone(data_matrix)
    if ordering == "leftright" or ordering == "position":
        orders = _orderByPosition(data_matrix)
    elif ordering == "weight":
        orders = _orderByWeight(data_matrix)
    elif ordering == "weightr":
        orders = _orderByWeightReverse(data_matrix)
    # mapped is a conversion of the values of the matrix in fracrion of the circumerence
    mapped = numpy.zeros(data_matrix.shape, dtype=float)
    dataLenght = data_matrix.shape[0]
    for j in range(dataLenght):
        mapped[:, j] = ideogram_length * data_matrix[:, j] / row_value
    ribbon_boundary = numpy.zeros((dataLenght, dataLenght, 2), dtype=float)
    for i in range(dataLenght):
        start = ideo_ends[i][0]
        # ribbon_boundary[i, orders[0]] = start
        for j in range(dataLenght):

            ribbon_boundary[i, orders[i, j], 0] = start
            ribbon_boundary[i, orders[i, j], 1] = start + mapped[i, orders[i, j]]

            start = ribbon_boundary[i, orders[i, j], 1]

    ribbons = []
    for i in range(dataLenght):
        for j in range(i + 1, dataLenght):
            if (
                data_matrix[i, j] < ignoreLessThan
                and data_matrix[j, i] < ignoreLessThan
            ):
                continue
            high, low = (i, j) if data_matrix[i, j] > data_matrix[j, i] else (j, i)

            ribbons.append(
                dict(
                    kind="flux" + ("ToZero" if data_matrix[low, high] == 0 else ""),
                    high=high,
                    low=low,
                    anglesHigh=(
                        ribbon_boundary[high, low, 0],
                        ribbon_boundary[high, low, 1],
                    ),
                    anglesLow=(
                        ribbon_boundary[low, high, 1],
                        ribbon_boundary[low, high, 0],
                    ),
                )
            )
    if not onlyFlux:
        for i in range(dataLenght):
            ribbons.append(
                dict(
                    kind="self",
                    id=i,
                    angles=(
                        ribbon_boundary[i, i, 0],
                        ribbon_boundary[i, i, 1],
                    ),
                )
            )

    return ribbons


def ChordDiagram(
    matrix: "list[list]",
    colors: Iterable = None,
    labels: list = None,
    ax: "plt.Axes|None" = None,
    GAP: float = numpy.rad2deg(2 * numpy.pi * 0.005),
    radius: float = 0.5,
    width: float = 0.05,
    ribbonposShift: float = 0.7,
    labelpos: float = 1.0,
    labelskwargs: dict = dict(),
    visualizationScale: float = 1.0,
    ignoreLessThan: int = 1,
    onlyFlux: bool = False,
    ordering: str = "matrix",
):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("off")
    ax.set_xlim(numpy.array([-1.0, 1.0]) * radius / visualizationScale)
    ax.set_ylim(numpy.array([-1.0, 1.0]) * radius / visualizationScale)
    ax.set_box_aspect(1)

    center = numpy.array([0.0, 0.0])
    wmatrix = numpy.array(matrix, dtype=int)
    row_sum, ideogram_length, ideo_ends = _prepareBases(wmatrix, GAP)
    myribbons = _ribbonCoordMaker(
        wmatrix,
        row_sum,
        ideogram_length,
        ideo_ends,
        ignoreLessThan=ignoreLessThan,
        onlyFlux=onlyFlux,
        ordering=ordering,
    )

    def getPos(x):
        return numpy.array([numpy.cos(numpy.deg2rad(x)), numpy.sin(numpy.deg2rad(x))])

    ribbonPos = radius - width * ribbonposShift
    FLUXPATH = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    FLUXTOZEROPATH = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    SELFPATH = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    for ribbon in myribbons:
        if ribbon["kind"] == "flux" or ribbon["kind"] == "fluxToZero":
            as1, as2 = ribbon["anglesHigh"]
            ae1, ae2 = ribbon["anglesLow"]
            s1 = center + ribbonPos * getPos(as1)
            s2 = center + ribbonPos * getPos(as2)
            s4, s3 = _bezierArcMaker(s2, s1, center)
            e1 = center + ribbonPos * getPos(ae1)
            e2 = center + ribbonPos * getPos(ae2)
            e3, e4 = _bezierArcMaker(e1, e2, center)
            ribbonPath = PathPatch(
                Path(
                    [s1, center, e1, e3, e4, e2, center, s2, s4, s3, s1, s1]
                    if ribbon["kind"] == "flux"
                    else [s1, center, e1, center, s2, s4, s3, s1, s1],
                    FLUXPATH if ribbon["kind"] == "flux" else FLUXTOZEROPATH,
                ),
                transform=ax.transData,
                alpha=0.4,
                zorder=1,
            )
            if colors is not None:
                ribbonPath.set(color=colors[ribbon["high"]])
            ax.add_patch(ribbonPath)
        elif ribbon["kind"] == "self":
            as1, as2 = ribbon["angles"]
            s1 = center + ribbonPos * getPos(as1)
            s2 = center + ribbonPos * getPos(as2)
            s4, s3 = _bezierArcMaker(s2, s1, center)

            ribbonPath = PathPatch(
                Path(
                    [s1, center, s2, s4, s3, s1, s1],
                    SELFPATH,
                ),
                # fc="none",
                transform=ax.transData,
                alpha=0.4,
                zorder=1,
            )
            if colors is not None:
                ribbonPath.set(color=colors[ribbon["id"]])
            ax.add_patch(ribbonPath)

    if width > 0:
        arcs = [Wedge(center, radius, a[0], a[1], width=width) for a in ideo_ends]

        p = PatchCollection(arcs, zorder=2)
        if colors is not None:
            p.set(color=colors)
        ax.add_collection(p)

    if labels:
        for i, a in enumerate(ideo_ends):
            pos = center + labelpos * radius * getPos(0.5 * (a[0] + a[1]))
            ax.text(pos[0], pos[1], labels[i], ha="center", va="center", **labelskwargs)
        


def prepareSOAP(trajFileName, trajAddress, rcut):
    soapFileName = trajFileName.split(".")[0] + "soap.hdf5"
    print(trajFileName, soapFileName)
    with h5py.File(trajFileName, "r") as workFile, h5py.File(
        soapFileName, "a"
    ) as soapFile:
        soapFile.require_group("SOAP")
        # skips if the soap trajectory is already present
        if trajAddress not in soapFile["SOAP"]:
            saponifyTrajectory(
                trajContainer=workFile[f"Trajectories/{trajAddress}"],
                SOAPoutContainer=soapFile["SOAP"],
                SOAPOutputChunkDim=200,
                SOAPnJobs=6,
                SOAPrcut=rcut,
                SOAPnmax=8,
                SOAPlmax=8,
            )
    return soapFileName

