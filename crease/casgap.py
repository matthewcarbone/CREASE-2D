from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from scipy.special import i1 as scipy_special_i1

from .common import CasgapParticleList, CasgapState
from .logger import logger
from .utils import Timer


def phaseI_construct_CasgapParticleList(state: CasgapState) -> CasgapState:
    params = state.params

    target_volfrac = params.volfrac
    box_vol = params.box_length**3
    vol_prefactor = 4 / 3 * np.pi
    mean_vol = vol_prefactor * params.mean_r**3

    # Total number of particles
    # Factor of 2 is for good measure.
    N = round(2 * target_volfrac / mean_vol * box_vol)

    # Parameters for lognormal distribution
    R_logmu = np.log(
        params.mean_r**2 / np.sqrt(params.mean_r**2 + params.sd_r**2)
    )
    gamma_logmu = np.log(
        params.mean_gamma**2
        / np.sqrt(params.mean_gamma**2 + params.sd_gamma**2)
    )
    R_logsigma = np.sqrt(np.log(1 + params.sd_r**2 / params.mean_r**2))
    gamma_logsigma = np.sqrt(
        np.log(1 + params.sd_gamma**2 / params.mean_gamma**2)
    )

    # Generate random samples of R
    Ri = lognormrandvar((N, 1), R_logmu, R_logsigma)
    partialvoli = vol_prefactor * Ri**3 / box_vol
    actual_volfrac = np.sum(partialvoli)

    while actual_volfrac <= target_volfrac:
        Ri_extra = lognormrandvar((N, 1), R_logmu, R_logsigma)
        Ri = np.concatenate((Ri, Ri_extra))
        voli_extra = vol_prefactor * Ri_extra**3 / box_vol
        partialvoli = np.concatenate((partialvoli, voli_extra))
        actual_volfrac = np.sum(partialvoli)

    if actual_volfrac > target_volfrac:
        N = np.argmax(
            np.cumsum(partialvoli) > target_volfrac
        )  # Find index where cumulative sum exceeds target_volfrac
        Ri = Ri[:N]

    # Generate random samples of gamma
    gammai = lognormrandvar((N, 1), gamma_logmu, gamma_logsigma)

    # Set spheroid axis lengths
    ai = Ri / gammai ** (1 / 3)
    ci = Ri * gammai ** (2 / 3)
    ac = np.hstack((ai, ci))
    Lambda = generate_quat(params.orientation_axis, params.omega)
    quat = samplequat(N, Lambda, params.kappa)

    state.particle_list = CasgapParticleList(
        n_particles=int(N),
        ac=ac,
        quat=quat,
        xyz=(np.random.rand(int(N), 3) - 0.5) * params.box_length,
    )

    logger.debug(f"Particle list summary: {state.particle_list}")

    return state


def phaseII(state: CasgapState) -> CasgapState:
    if state.particle_list is None:
        raise ValueError("particle_list should not be None in phaseII")

    if state.particle_list.n_prime:
        # Check and adjust coords so that the new ellipsoid does not overlap
        # with any previous ellipsoids.
        successflag, coords = check_overlap(state)
        if not successflag:
            state.in_error_state = True
            return state

        # Increment population
        population = state.particle_list.n_prime + 1
        state.particle_list.xyz[population - 1, :] = coords
    else:
        # It is the first particle, there can't be any overlap!
        population = 1
        coords = state.particle_list.xyz[population - 1, :]

    # Append polyhedra
    axislengths = state.particle_list.ac[population - 1, :]
    quats = state.particle_list.quat[population - 1, :]
    polyhedron = discretize_ellipsoid(axislengths, quats, coords)
    state.particle_list.polyhedra[population - 1] = polyhedron
    state.particle_list.n_prime = population
    return state


def casgap(state: CasgapState):
    np.random.seed(state.params.seed)

    # Intelligently make use of checkpoints
    # In this case, we only need to re-run phaseI if the particle_list
    # does not exist
    if state.particle_list is None:
        with Timer() as timer:
            state = phaseI_construct_CasgapParticleList(state)
        logger.info(f"phaseI complete in {timer.elapsed:.02f} s")
    if state.particle_list is None:
        raise ValueError("Something went terribly wrong: particle_list is None")

    # At this point, we need to restart the phaseII calculation

    with Timer() as outer_timer:
        start = state.params.phaseII_loop_start
        for i in range(start, state.particle_list.n_particles + 1):
            np.random.seed(state.params.seed + i)

            with Timer() as inner_timer:
                state = phaseII(state)
            logger.debug(f"pop={i} complete in {inner_timer.elapsed:.02f} s")

            if state.in_error_state:
                break

            if i % state.params.checkpoint_frequency == 0:
                state.params.phaseII_loop_start = i + 1
                state.checkpoint()
                logger.info(f"Checkpoint at population {i}")

    logger.info(f"All phaseII loops complete in {outer_timer.elapsed:.02f} s")

    # Final checkpoint when done
    state.checkpoint()

    if state.particle_list is None:
        raise ValueError("particle_list should not be None")

    a = state.particle_list.ac[: state.particle_list.n_prime, 0]
    b = state.particle_list.ac[: state.particle_list.n_prime, 1]
    partialvoli = 4.0 / 3.0 * np.pi * a**2 * b

    params = state.params

    actual_volfrac = np.sum(partialvoli) / params.box_length**3

    if state.in_error_state:
        logger.error(
            f"Not all ellipsoids could be added. Final population is {state.particle_list.n_prime}. Final volume fraction is {actual_volfrac:.05f}."
        )
    else:
        logger.success(
            f"All ellipsoids were successfully added! Final population is {state.particle_list.n_prime}. Final volume fraction is {actual_volfrac:.05f}."
        )


def check_overlap(state: CasgapState) -> Tuple[int, np.ndarray]:
    successflag = 0

    particle_list = state.particle_list

    if particle_list is None:
        raise ValueError("particle_list should not be None in check_overlap")

    params = state.params

    population = particle_list.n_prime
    newind = population + 1
    ac = particle_list.ac[newind - 1]
    q = particle_list.quat[newind - 1]
    boundingradius = np.max(ac)
    newpolyhedra_unshifted = discretize_ellipsoid(ac, q, [0, 0, 0])

    maxattempts = 10000
    perturbfreq = 100  # resample new coords at this rate
    maxradii = np.max(particle_list.ac[:population], axis=1)
    pos = particle_list.xyz[:population]

    EPAclearance = 0.1
    for numattempts in range(1, maxattempts + 1):
        if numattempts % perturbfreq == 1:
            if numattempts < perturbfreq:
                coords = (np.random.rand(3) - 0.5) * params.box_length
            else:
                randvars = np.random.rand(3)
                zbins = np.linspace(
                    -0.5 * params.box_length, 0.5 * params.box_length, 11
                )
                znum, _ = np.histogram(pos[:, 2], bins=zbins)
                maxznum = np.max(znum)
                minznum = np.min(znum)
                if maxznum == minznum:
                    zcoord = (randvars[0] - 0.5) * params.box_length
                else:
                    newznum = (maxznum - znum) / (maxznum - minznum)
                    cumznum = (
                        np.concatenate(([0], np.cumsum(newznum)))
                        + np.arange(11) * 1e-10
                    )
                    cumznum /= cumznum[-1]
                    zcoord = np.interp(randvars[2], cumznum, zbins)
                zbinwidth = zbins[1] - zbins[0]
                zfilter = (pos[:, 2] > zcoord - zbinwidth / 2) & (
                    pos[:, 2] <= zcoord + zbinwidth / 2
                )

                ybins = np.linspace(
                    -0.5 * params.box_length, 0.5 * params.box_length, 11
                )
                ynum, _ = np.histogram(pos[zfilter, 1], bins=ybins)
                maxynum = np.max(ynum)
                minynum = np.min(ynum)
                if not np.sum(zfilter) or maxynum == minynum:
                    ycoord = (randvars[1] - 0.5) * params.box_length
                else:
                    newynum = (maxynum - ynum) / (maxynum - minynum)
                    cumynum = (
                        np.concatenate(([0], np.cumsum(newynum)))
                        + np.arange(11) * 1e-10
                    )
                    cumynum /= cumynum[-1]
                    ycoord = np.interp(randvars[1], cumynum, ybins)
                ybinwidth = ybins[1] - ybins[0]
                yzfilter = (
                    zfilter
                    & (pos[:, 1] > ycoord - ybinwidth / 2)
                    & (pos[:, 1] <= ycoord + ybinwidth / 2)
                )

                xbins = np.linspace(
                    -0.5 * params.box_length, 0.5 * params.box_length, 11
                )
                xnum, _ = np.histogram(pos[yzfilter, 0], bins=xbins)
                maxxnum = np.max(xnum)
                minxnum = np.min(xnum)
                if not np.sum(yzfilter) or maxxnum == minxnum:
                    xcoord = (randvars[0] - 0.5) * params.box_length
                else:
                    newxnum = (maxxnum - xnum) / (maxxnum - minxnum)
                    cumxnum = (
                        np.concatenate(([0], np.cumsum(newxnum)))
                        + np.arange(11) * 1e-10
                    )
                    cumxnum /= cumxnum[-1]
                    xcoord = np.interp(randvars[0], cumxnum, xbins)
                coords = np.array([xcoord, ycoord, zcoord])

        newpolyhedra = newpolyhedra_unshifted.copy()
        newpolyhedra["vertices"] = (
            newpolyhedra["vertices"]
            + np.ones((newpolyhedra["vertices"].shape[0], 1)) * coords
        )
        dist = np.sqrt(
            np.sum((np.ones((population, 1)) * coords - pos) ** 2, axis=1)
        )
        potentialoverlap = maxradii + boundingradius - dist > 0

        if not np.any(potentialoverlap):
            successflag = 1
            break

        indices = np.where(potentialoverlap)[0]
        numintersections = 0
        cummulativeshift = np.array([0, 0, 0])
        for i in indices:
            # try:
            intersectionflag, intersectionsimplex = gjk_simplex(
                newpolyhedra, particle_list.polyhedra[i]
            )
            # except KeyError:
            #    continue

            if intersectionflag:
                numintersections += 1
                shiftvector, shiftdist = expandingpolytope_shift(
                    newpolyhedra,
                    particle_list.polyhedra[i],
                    intersectionsimplex,
                )
                cummulativeshift = -shiftvector * (shiftdist + EPAclearance)

        if np.any(cummulativeshift):
            if not numattempts % 100:
                print(
                    f"Warning: Previous attempts failed to converge after {numattempts} attempts. Current ellipsoid has {numintersections} intersections. The population is {population}."
                )
            coords = coords + cummulativeshift
            coords = (
                np.mod(coords + params.box_length / 2, params.box_length)
                - params.box_length / 2
            )
        else:
            successflag = 1
            break

    return successflag, coords


def discretize_ellipsoid(axislengths, q, origin):
    # Find the axial vector from the quaternion
    axvec = np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] + q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ],
        ]
    )
    mag_axvec = np.sqrt(axvec[0] ** 2 + axvec[1] ** 2 + axvec[2] ** 2)
    axvec = axvec / mag_axvec

    # Convert Ellipsoid to Polyhedron
    f = np.sqrt(2) - 1
    a, b, c = axislengths[0], axislengths[0], axislengths[1]
    X0 = a * np.array([1, f, f])
    Y0 = b * np.array([f, 1, f])
    Z0 = c * np.array([f, f, 1])
    allvertices = np.concatenate(
        [
            np.column_stack((X0, Y0, Z0)),  # first octant: x y z
            np.column_stack((-X0, Y0, Z0)),  # second octant: -x y z
            np.column_stack((-X0, -Y0, Z0)),  # third octant: -x -y z
            np.column_stack((X0, -Y0, Z0)),  # fourth octant: x -y z
            np.column_stack((X0, Y0, -Z0)),  # fifth octant: x y -z
            np.column_stack((-X0, Y0, -Z0)),  # sixth octant: -x y -z
            np.column_stack((-X0, -Y0, -Z0)),  # seventh octant: -x -y -z
            np.column_stack((X0, -Y0, -Z0)),  # eighth octant: x -y -z
        ]
    )
    numvertices = allvertices.shape[0]
    allvertices = (
        np.dot(allvertices, axvec.T) + np.ones((numvertices, 1)) * origin
    )
    allfaces = [
        [1, 10, 22, 13],  # +x rect
        [4, 16, 19, 7],  # -x rect
        [2, 14, 17, 5],  # +y rect
        [8, 20, 23, 11],  # -y rect
        [3, 6, 9, 12],  # +z rect
        [15, 24, 21, 18],  # -z rect
        [1, 3, 12, 10],  # +x+z rect
        [4, 7, 9, 6],  # -x+z rect
        [16, 18, 21, 19],  # -x-z rect
        [13, 22, 24, 15],  # +x-z rect
        [2, 5, 6, 3],  # +y+z rect
        [8, 11, 12, 9],  # -y+z rect
        [20, 21, 24, 23],  # -y-z rect
        [14, 15, 18, 17],  # +y-z rect
        [1, 13, 14, 2],  # +x+y rect
        [4, 5, 17, 16],  # -x+y rect
        [7, 19, 20, 8],  # -x-y rect
        [10, 11, 23, 22],  # +x-y rect
        [1, 2, 3],  # +x+y+z tri
        [4, 6, 5],  # -x+y+z tri
        [7, 8, 9],  # -x-y+z tri
        [10, 12, 11],  # +x-y+z tri
        [13, 15, 14],  # +x+y-z tri
        [16, 17, 18],  # -x+y-z tri
        [19, 21, 20],  # -x-y-z tri
        [22, 23, 24],  # +x-y-z tri
    ]
    polyhedra = {"vertices": allvertices, "faces": allfaces, "center": origin}

    return polyhedra


def expandingpolytope_shift(polyhedron1, polyhedron2, simplex):
    polytope = {"vertices": simplex["vertices"], "faces": simplex["faces"]}
    facenorms = getfacenormals(polytope)
    tolerance = 1e-6
    # counter=0
    while True:
        # counter += 1
        # print(f"I am stuck here. {counter}")
        minInd, mindist = findnearestface(polytope, facenorms)
        nearestnormal = facenorms[minInd]
        support = supportpoint(polyhedron1, polyhedron2, nearestnormal)[0]
        supportdist = np.dot(nearestnormal, support)
        if (supportdist - mindist) > tolerance:
            numnormals = facenorms.shape[0]
            uniqueedges = []
            newfaces = []
            newnormals = []
            for i in range(numnormals):
                face = polytope["faces"][i]
                facevertex1 = polytope["vertices"][face[0] - 1]
                if np.dot(facenorms[i], support - facevertex1) > tolerance:
                    faceedges = np.column_stack((face[:-1], face[1:]))
                    faceedges = np.vstack((faceedges, [face[-1], face[0]]))
                    uniqueedges = finduniqueedges(uniqueedges, faceedges)
                else:
                    newfaces.append(polytope["faces"][i])
                    newnormals.append(facenorms[i])
            # add the new support point to the polytope
            polytope["vertices"] = np.vstack((polytope["vertices"], support))
            supportind = polytope["vertices"].shape[0]
            numuniqueedges = uniqueedges.shape[0]
            for i in range(numuniqueedges):
                newaddedface, newaddedfacenorm = createnewpolytopeface(
                    polytope, uniqueedges[i], supportind
                )
                newfaces.append(newaddedface)
                newnormals.append(newaddedfacenorm)
            polytope["faces"] = newfaces
            facenorms = np.array(newnormals)
        else:
            shiftvector = nearestnormal
            shiftdist = mindist
            break

    return shiftvector, shiftdist


def getfacenormals(polytope):
    faces = polytope["faces"]
    numfaces = len(faces)
    facenorms = np.zeros((numfaces, 3))
    tolerance = 1e-6
    for i in range(numfaces):
        face = faces[i]
        point_a = polytope["vertices"][face[0] - 1]
        point_b = polytope["vertices"][face[1] - 1]
        point_c = polytope["vertices"][face[2] - 1]
        vec_ab = point_b - point_a
        vec_bc = point_c - point_b
        facenorm_abc = np.cross(vec_ab, vec_bc)
        facedist = np.dot(point_a, facenorm_abc)
        if facedist < tolerance:
            facenorm_abc = -facenorm_abc
        facenorms[i] = facenorm_abc / np.linalg.norm(
            facenorm_abc
        )  # These are unit vectors

    return facenorms


def finduniqueedges(uniqueedges, faceedges):
    if len(uniqueedges) == 0:
        uniqueedges = faceedges
    else:
        for i in range(faceedges.shape[0]):
            # check if the reverse of  uniqueedge is in the faceedge. If there is, then delete.
            uniqueedge_vert1 = uniqueedges[:, 0] - faceedges[i, 1]
            uniqueedge_vert2 = uniqueedges[:, 1] - faceedges[i, 0]
            uniqueness_check = np.logical_not(
                uniqueedge_vert1
            ) & np.logical_not(uniqueedge_vert2)
            if np.any(uniqueness_check):
                uniqueedges = uniqueedges[np.logical_not(uniqueness_check), :]
            else:
                uniqueedges = np.vstack((uniqueedges, faceedges[i, :]))
    return uniqueedges


def createnewpolytopeface(polytope, edge, pointind):
    point_a = polytope["vertices"][edge[0] - 1, :]
    point_b = polytope["vertices"][edge[1] - 1, :]
    point_c = polytope["vertices"][pointind - 1, :]
    tolerance = 1e-6
    vec_ab = point_b - point_a
    vec_bc = point_c - point_b
    facenorm_abc = np.cross(vec_ab, vec_bc)
    facenorm = facenorm_abc / np.linalg.norm(
        facenorm_abc
    )  # These are unit vectors
    facedist = np.dot(point_a, facenorm)
    if facedist > tolerance:
        face = [edge[0], edge[1], pointind]
    else:
        face = [edge[1], edge[0], pointind]
        facenorm = -facenorm
    return face, facenorm


def findnearestface(polytope, facenorms):
    # distances = np.dot(polytope['vertices'][x[0]-1 for x in polytope['faces']], facenorms.T)
    facevertex = np.array(
        [polytope["vertices"][x[0] - 1] for x in polytope["faces"]]
    )
    distances = np.einsum("ij,ij->i", facevertex, facenorms)
    minInd = np.argmin(distances)
    mindist = distances[minInd]
    return minInd, mindist


def gjk_simplex(polyhedron1, polyhedron2):
    direction = polyhedron1["center"] - polyhedron2["center"]
    support = supportpoint(polyhedron1, polyhedron2, direction)[0]
    simplexvertices = support
    direction = -np.array(support)

    while True:
        support = supportpoint(polyhedron1, polyhedron2, direction)[0]

        if np.all(np.logical_not(direction)) or np.dot(support, direction) <= 0:
            intersection_flag = 0
            simplex = {"vertices": []}
            break

        simplexvertices = np.vstack((support, simplexvertices))
        flag, simplexvertices, direction = nextsimplex(simplexvertices)

        if flag:
            if simplexvertices.shape[0] < 4:
                intersection_flag = 0
                simplex = {"vertices": []}
            else:
                intersection_flag = 1
                simplexfaces = gettetrahedronfaces(simplexvertices, [0, 0, 0])
                simplex = {"vertices": simplexvertices, "faces": simplexfaces}
            break

    return intersection_flag, simplex


def nextsimplex(vertices):
    num_vertices = len(vertices)
    if num_vertices == 2:
        flag, vertices, direction = linesimplex(vertices)
    elif num_vertices == 3:
        flag, vertices, direction = trianglesimplex(vertices)
    elif num_vertices == 4:
        flag, vertices, direction = tetrahedronsimplex(vertices)
    else:
        flag = False
        vertices = []
        direction = np.array([])

    return flag, vertices, direction


def linesimplex(vertices):
    direction = np.zeros(3)
    point_a = vertices[0]
    point_b = vertices[1]
    vec_ab = point_b - point_a
    vec_ao = -point_a
    norm_abo = np.cross(vec_ab, vec_ao)

    if np.linalg.norm(norm_abo):  # check for collinearity
        direction = np.cross(norm_abo, vec_ab)
        flag = False
    else:
        flag = True

    return flag, vertices, direction


def trianglesimplex(vertices):
    direction = np.zeros(3)
    point_a = vertices[0]
    point_b = vertices[1]
    point_c = vertices[2]

    vec_ab = point_b - point_a
    vec_ac = point_c - point_a
    vec_ao = -point_a

    facenorm_abc = np.cross(vec_ab, vec_ac)

    if np.linalg.norm(facenorm_abc):  # check if origin is coplanar
        if np.dot(facenorm_abc, vec_ao) > 0:
            direction = facenorm_abc
            flag = False
        else:
            direction = -facenorm_abc
            flag = False
    else:
        flag = True

    return flag, vertices, direction


def tetrahedronsimplex(vertices):
    direction = np.zeros(3)
    point_a = vertices[0]
    point_b = vertices[1]
    point_c = vertices[2]
    point_d = vertices[3]

    vec_ab = point_b - point_a
    vec_ac = point_c - point_a
    vec_ad = point_d - point_a
    vec_ao = -point_a

    facenorm_abc = np.cross(vec_ab, vec_ac)
    if np.dot(facenorm_abc, vec_ad) > 0:
        facenorm_abc = -facenorm_abc

    facenorm_acd = np.cross(vec_ac, vec_ad)
    if np.dot(facenorm_acd, vec_ab) > 0:
        facenorm_acd = -facenorm_acd

    facenorm_abd = np.cross(vec_ab, vec_ad)
    if np.dot(facenorm_abd, vec_ac) > 0:
        facenorm_abd = -facenorm_abd

    outsideabc = np.dot(facenorm_abc, vec_ao) > 0
    outsideacd = np.dot(facenorm_acd, vec_ao) > 0
    outsideabd = np.dot(facenorm_abd, vec_ao) > 0

    if (
        (outsideabc and outsideacd)
        or (outsideacd and outsideabd)
        or (outsideabd and outsideabc)
    ):
        flag = False
    elif outsideabc:
        vertices = np.array([point_a, point_b, point_c])
        direction = facenorm_abc
        flag = False
    elif outsideacd:
        vertices = np.array([point_a, point_c, point_d])
        direction = facenorm_acd
        flag = False
    elif outsideabd:
        vertices = np.array([point_a, point_b, point_d])
        direction = facenorm_abd
        flag = False
    else:
        flag = True

    return flag, vertices, direction


def gettetrahedronfaces(vertices, interiorpoint):
    if vertices.shape[0] != 4:
        raise ValueError("Input should be a set of tetrahedral vertices")

    faces = [[1, 2, 3], [1, 4, 2], [1, 3, 4], [2, 4, 3]]

    for i in range(4):
        face = faces[i]
        point_a = vertices[face[0] - 1]
        point_b = vertices[face[1] - 1]
        point_c = vertices[face[2] - 1]

        vec_ab = point_b - point_a
        vec_bc = point_c - point_b
        facenorm_abc = np.cross(vec_ab, vec_bc)

        if np.dot(facenorm_abc, point_b - interiorpoint) < 0:
            faces[i] = [face[0], face[2], face[1]]

    return faces


def lognormrandvar(size, logmu, logsigma):
    temp = -np.sqrt(2) * erfcinv(
        2 * np.random.rand(*size)
    )  # Std normal random variable
    result = np.exp(logmu + temp * logsigma)
    return result


def generate_quat(W, omega):
    alpha_by_2 = omega / 2 * np.pi / 180  # Convert angle to radians
    norm = W / np.sqrt(np.sum(W**2))
    q = np.array(
        [
            np.cos(alpha_by_2),
            norm[0] * np.sin(alpha_by_2),
            norm[1] * np.sin(alpha_by_2),
            norm[2] * np.sin(alpha_by_2),
        ]
    )
    return q


def samplequat(num, pref_q, kappa):
    v = -np.sqrt(2) * erfcinv(2 * np.random.rand(num, 2))
    vsum = np.sqrt(np.sum(v**2, axis=1))
    v = v / vsum[:, np.newaxis]  # Ensure shape compatibility

    if kappa:
        randnum = np.random.rand(num)
        w = 1 + 1 / kappa * np.log(randnum + (1 - randnum) * np.exp(-2 * kappa))
    else:
        w = 2 * np.random.rand(num) - 1

    orgvec = np.hstack((np.zeros((num, 2)), np.ones((num, 1))))
    newvec = np.vstack(
        (w, np.sqrt(1 - w**2) * v[:, 0], np.sqrt(1 - w**2) * v[:, 1])
    ).T

    axisvec = np.cross(orgvec, newvec)
    axisvec /= np.sqrt(
        np.sum(axisvec**2, axis=1, keepdims=True)
    )  # Ensure shape compatibility
    axistheta = np.arccos(np.sum(orgvec * newvec, axis=1))

    cos_half = np.cos(axistheta / 2)[:, np.newaxis]  # Reshape to (num, 1)
    sin_half = np.sin(axistheta / 2)[:, np.newaxis]  # Reshape to (num, 1)
    quats = np.hstack(
        (cos_half, sin_half * axisvec)
    )  # Concatenate along axis 1

    if kappa:
        Mumat = np.zeros((4, 4))
        Mumat[:, 0] = pref_q.T
        Q, R = np.linalg.qr(Mumat)
        if R[0, 0] < 0:
            Q = -Q
        result = np.dot(Q, quats.T).T
    else:
        result = quats

    return result


def samplew(size, kappa):
    u = np.random.rand(size)
    w = np.arange(-1, 1, 0.0001)
    first_two_terms = 1 + (w * np.sqrt(1 - w**2) - np.arccos(w)) / np.pi
    mplus2_term = (
        lambda m: (m + 1)
        / np.pi
        * (
            np.sin((m + 2) * np.arccos(w)) / (m + 2)
            - np.sin(m * np.arccos(w)) / m
        )
        * scipy_special_i1(m + 1, kappa)
        / scipy_special_i1(1, kappa)
    )
    Fw = first_two_terms
    mvalue = 1
    if kappa:
        while True:
            newterm = mplus2_term(mvalue)
            newterm_magnitude = np.sqrt(np.sum(newterm**2))
            if np.isnan(newterm_magnitude):
                raise ValueError("Nan values detected, kappa is too high!")
            elif abs(newterm_magnitude) < 1e-10:
                break
            Fw += newterm
            mvalue += 1
    Fw[np.abs(Fw) < 1e-10] = 0
    if not np.all(np.diff(Fw) >= 0):
        raise ValueError("Cumulative function is not monotonic!")
    smallseries = np.linspace(0, 1e-15, len(w))
    data = np.column_stack((w, (Fw + smallseries) / (Fw[-1] + smallseries[-1])))
    sampledvalues = interp1d(data[:, 1], data[:, 0])(u)
    return sampledvalues


def supportpoint(polyhedron1, polyhedron2, direction):
    furthestpoint1 = furthestpoint(polyhedron1["vertices"], direction)
    furthestpoint2 = furthestpoint(polyhedron2["vertices"], -direction)
    point = furthestpoint1 - furthestpoint2
    return point, furthestpoint1, furthestpoint2


def furthestpoint(vertices, direction):
    maxpoint = np.zeros(3)
    maxprojection = -np.inf
    for i in range(vertices.shape[0]):
        projection = np.dot(vertices[i], direction)
        if maxprojection < projection:
            maxprojection = projection
            maxpoint = vertices[i]
    return maxpoint
