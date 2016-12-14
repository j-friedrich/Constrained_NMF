# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components,
    given the spatial components and temporal components
@author: agiovann, j-friedrich
"""
from scipy.sparse import spdiags, coo_matrix
import scipy
import numpy as np
from deconvolution import constrained_foopsi
from utilities import update_order
import sys


def make_G_matrix(T, g):
    ''' create matrix of autoregression to enforce indicator dynamics
    Inputs:
    T: positive integer
        number of time-bins
    g: nd.array, vector p x 1
        Discrete time constants

    Output:
    G: sparse diagonal matrix
        Matrix of autoregression
    '''
    if type(g) is np.ndarray:
        if len(g) == 1 and g < 0:
            g = 0
        gs = np.matrix(np.hstack((1, -(g[:]).T)))
        ones_ = np.matrix(np.ones((T, 1)))
        G = spdiags((ones_ * gs).T, range(0, -len(g) - 1, -1), T, T)
        return G
    else:
        raise Exception('g must be an array')


def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi
    """

    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T = np.shape(Ytemp)[0]
    cc_, cb_, c1_, gn_, sn_, sp_ = constrained_foopsi(Ytemp, bl=bl, c1=c1, g=g, sn=sn, **argss)
    if cc_ is not None:
        gd_ = np.max(np.roots(np.hstack((1, -gn_.T))))
        gd_vec = gd_**range(T)

        C_ = cc_[:].T + cb_ + np.dot(c1_, gd_vec)
        Sp_ = sp_[:T].T
        Ytemp_ = Ytemp - C_.T
    else:
        C_ = None
        Sp_ = None
        Ytemp_ = None

    return C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_


def update_temporal_components(Y, A, b, Cin, fin, bl=None, c1=None, g=None, sn=None,
                               nb=1, ITER=2, method_foopsi='constrained_foopsi',
                               memory_efficient=False, debug=False, dview=None, **kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.
    Parameters
    -----------

    Y: np.ndarray (2D)
        input data with time in the last axis (d x T)
    A: sparse matrix (crc format)
        matrix of temporal components (d x K)
    b: ndarray (dx1)
        current estimate of background component
    Cin: np.ndarray
        current estimate of temporal components (K x T)
    fin: np.ndarray
        current estimate of temporal background (vector of length T)
    g:  np.ndarray
        Global time constant (not used)
    bl: np.ndarray
       baseline for fluorescence trace for each column in A
    c1: np.ndarray
       initial concentration for each column in A
    g:  np.ndarray
       discrete time constant for each column in A
    sn: np.ndarray
       noise level for each column in A
    nb: [optional] int
        Number of background components
    ITER: positive integer
        Maximum number of block coordinate descent loops.
    method_foopsi: string
        Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.
    n_processes: int
        number of processes to use for parallel computation. Should be less than the number of processes started with ipcluster.
    backend: 'str'
        single_thread no parallelization
        ipyparallel, parallelization using the ipyparallel cluster. You should start the cluster (install ipyparallel and then type 
        ipcluster -n 6, where 6 is the number of processes).
        SLURM: using SLURM scheduler
    memory_efficient: Bool
        whether or not to optimize for memory usage (longer running times). nevessary with very large datasets  
    **kwargs: dict
        all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation). Some useful parameters are      
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for constrained foopsi. Choices are
            'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
            'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note
    --------

    The temporal components are updated in parallel by default by forming of sequence of vertex covers.  

    Returns
    --------

    C:   np.ndarray
            matrix of temporal components (K x T)
    f:   np.array
            vector of temporal background (length T) 
    S:   np.ndarray            
            matrix of merged deconvolved activity (spikes) (K x T)
    bl:  float  
            same as input    
    c1:  float
            same as input    
    g:   float
            same as input    
    sn:  float
            same as input 
    YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.            
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)

    """

    if not kwargs.has_key('p') or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    d, T = np.shape(Y)
    nr = np.shape(A)[-1]

    if b is not None:
        if b.shape[0] < b.shape[1]:
            b = b.T
        nb = b.shape[1]

    if bl is None:
        bl = np.repeat(None, nr)

    if c1 is None:
        c1 = np.repeat(None, nr)

    if g is None:
        g = np.repeat(None, nr)

    if sn is None:
        sn = np.repeat(None, nr)

    A = scipy.sparse.hstack((A, coo_matrix(b)))
    S = np.zeros(np.shape(Cin))
    Cin = np.vstack((Cin, fin))
    C = Cin
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()), axis=0)))

    Cin = coo_matrix(Cin)
    YA = (A.T.dot(Y).T) * spdiags(1. / nA, 0, nr + nb, nr + nb)
    AA = ((A.T.dot(A)) * spdiags(1. / nA, 0, nr + nb, nr + nb)).tocsr()
    YrA = YA - Cin.T.dot(AA)

    Cin = np.array(Cin.todense())
    for iter in range(ITER):
        O, lo = update_order(A.tocsc()[:, :nr])
        P_ = []
        for count, jo_ in enumerate(O):
            jo = np.array(list(jo_))
            Ytemp = YrA[:, jo.flatten()] + Cin[jo, :].T
            Ctemp = np.zeros((np.size(jo), T))
            Stemp = np.zeros((np.size(jo), T))
            # btemp = np.zeros((np.size(jo), 1))
            # sntemp = btemp.copy()
            # c1temp = btemp.copy()
            gtemp = np.zeros((np.size(jo), kwargs['p']))
            nT = nA[jo]

            args_in = [(np.squeeze(np.array(Ytemp[:, jj])), nT[jj], jj, None,
                        None, None, None, kwargs) for jj in range(len(jo))]
            if dview is not None:
                if debug:
                    results = dview.map_async(constrained_foopsi_parallel, args_in)
                    results.get()
                    for outp in results.stdout:
                        print outp[:-1]
                        sys.stdout.flush()
                    for outp in results.stderr:
                        print outp[:-1]
                        sys.stderr.flush()
                else:
                    results = dview.map_sync(constrained_foopsi_parallel, args_in)

            else:
                results = map(constrained_foopsi_parallel, args_in)

            for chunk in results:

                pars = dict()
                C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_ = chunk
                Ctemp[jj_, :] = C_[None, :]
                Stemp[jj_, :] = Sp_
                Ytemp[:, jj_] = Ytemp_[:, None]
                # btemp[jj_] = cb_
                # c1temp[jj_] = c1_
                # sntemp[jj_] = sn_
                gtemp[jj_, :] = gn_.T
                bl[jo[jj_]] = cb_
                c1[jo[jj_]] = c1_
                sn[jo[jj_]] = sn_
                g[jo[jj_]] = gn_.T if kwargs['p'] > 0 else []  # gtemp[jj,:]
                pars['b'] = cb_
                pars['c1'] = c1_
                pars['neuron_sn'] = sn_
                pars['gn'] = gtemp[jj_, np.abs(gtemp[jj, :]) > 0]
                pars['neuron_id'] = jo[jj_]
                P_.append(pars)

            YrA -= (Ctemp - C[jo, :]).T * AA[jo, :]
            C[jo, :] = Ctemp.copy()
            S[jo, :] = Stemp

            print str(np.sum(lo[:count + 1])) + ' out of total ' + str(nr) + ' temporal components updated'

        for ii in np.arange(nr, nr + nb):
            cc = np.maximum(YrA[:, ii] + np.atleast_2d(Cin[ii, :]).T, 0)
            YrA -= (cc - np.atleast_2d(Cin[ii, :]).T) * AA[ii, :]
            C[ii, :] = cc.T

        if dview is not None:
            dview.results.clear()

        if scipy.linalg.norm(Cin - C, 'fro') / scipy.linalg.norm(C, 'fro') <= 1e-3:
            print "stopping: overall temporal component not changing significantly"
            break
        else:
            Cin = C

    f = C[nr:, :]
    C = C[:nr, :]
    YrA = np.array(YrA[:, :nr]).T
    P_ = sorted(P_, key=lambda k: k['neuron_id'])

    return C, f, S, bl, c1, sn, g, YrA


def update_temporal_components_interleaved(Y, A, b, Cin, fin, dims, ds=1, bl=None, c1=None, g=None, sn=None,
                                           nb=1, ITER=2, method_foopsi='constrained_foopsi',
                                           memory_efficient=False, debug=False, dview=None, **kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.
    Parameters
    -----------

    Y: np.ndarray (2D)
        input data with time in the last axis (d x T)
    A: sparse matrix (crc format)
        matrix of temporal components (d x K)
    b: ndarray (dx1)
        current estimate of background component
    Cin: np.ndarray
        current estimate of temporal components (K x T)
    fin: np.ndarray
        current estimate of temporal background (vector of length T)
    g:  np.ndarray
        Global time constant (not used)
    bl: np.ndarray
       baseline for fluorescence trace for each column in A
    c1: np.ndarray
       initial concentration for each column in A
    g:  np.ndarray
       discrete time constant for each column in A
    sn: np.ndarray
       noise level for each column in A
    nb: [optional] int
        Number of background components
    ITER: positive integer
        Maximum number of block coordinate descent loops.
    method_foopsi: string
        Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.
    n_processes: int
        number of processes to use for parallel computation. Should be less than the number of processes started with ipcluster.
    backend: 'str'
        single_thread no parallelization
        ipyparallel, parallelization using the ipyparallel cluster. You should start the cluster (install ipyparallel and then type 
        ipcluster -n 6, where 6 is the number of processes).
        SLURM: using SLURM scheduler
    memory_efficient: Bool
        whether or not to optimize for memory usage (longer running times). nevessary with very large datasets  
    **kwargs: dict
        all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation). Some useful parameters are      
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for constrained foopsi. Choices are
            'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
            'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note
    --------

    The temporal components are updated in parallel by default by forming of sequence of vertex covers.  

    Returns
    --------

    C:   np.ndarray
            matrix of temporal components (K x T)
    f:   np.array
            vector of temporal background (length T)
    S:   np.ndarray
            matrix of merged deconvolved activity (spikes) (K x T)
    bl:  float
            same as input
    c1:  float
            same as input
    g:   float
            same as input
    sn:  float
            same as input
    YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.            
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)

    """

    if not kwargs.has_key('p') or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    d, T = np.shape(Y)
    nr = np.shape(A)[-1]

    if b is not None:
        if b.shape[0] < b.shape[1]:
            b = b.T
        nb = b.shape[1]

    if bl is None:
        bl = np.repeat(None, nr)

    if c1 is None:
        c1 = np.repeat(None, nr)

    if g is None:
        g = np.repeat(None, nr)

    if sn is None:
        sn = np.repeat(None, nr)

    A = scipy.sparse.hstack((A, coo_matrix(b)))
    S = np.zeros(np.shape(Cin))
    Cin = np.vstack((Cin, fin))
    C = Cin.copy()
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()), axis=0)))
    Aeven = standardDownscale(A.toarray().reshape(
        tuple(dims) + (-1,)), ds).reshape(-1, nr + nb)
    Aodd = shiftedDownscale(A.toarray().reshape(
        tuple(dims) + (-1,)), ds).reshape(-1, nr + nb)
    nAeven = np.squeeze(np.array(np.sum(np.square(Aeven), axis=0)))
    nAodd = np.squeeze(np.array(np.sum(np.square(Aodd), axis=0)))
    YAeven = (Aeven.T.dot(Y[:, ::2]).T) * spdiags(1. / nAeven, 0, nr + nb, nr + nb)
    YAodd = (Aodd.T.dot(Y[:, 1::2]).T) * spdiags(1. / nAodd, 0, nr + nb, nr + nb)
    AAeven = ((Aeven.T.dot(Aeven)) * spdiags(1. / nAeven, 0, nr + nb, nr + nb))
    AAodd = ((Aodd.T.dot(Aodd)) * spdiags(1. / nAodd, 0, nr + nb, nr + nb))
    YrAeven = YAeven - Cin.T[::2].dot(AAeven)
    YrAodd = YAodd - Cin.T[1::2].dot(AAodd)
    YrA = np.matrix(np.zeros((T, nr + nb)))
    YrA[::2] = YrAeven
    YrA[1::2] = YrAodd

    for iter in range(ITER):
        O, lo = update_order(A.tocsc()[:, :nr])
        P_ = []
        for count, jo_ in enumerate(O):
            jo = np.array(list(jo_))
            Ytemp = YrA[:, jo.flatten()] + Cin[jo, :].T
            Ctemp = np.zeros((np.size(jo), T))
            Stemp = np.zeros((np.size(jo), T))
            # btemp = np.zeros((np.size(jo), 1))
            # sntemp = btemp.copy()
            # c1temp = btemp.copy()
            gtemp = np.zeros((np.size(jo), kwargs['p']))
            nT = nA[jo]

            args_in = [(np.squeeze(np.array(Ytemp[:, jj])), nT[jj], jj, None,
                        None, None, None, kwargs) for jj in range(len(jo))]
            if dview is not None:
                if debug:
                    results = dview.map_async(constrained_foopsi_parallel, args_in)
                    results.get()
                    for outp in results.stdout:
                        print outp[:-1]
                        sys.stdout.flush()
                    for outp in results.stderr:
                        print outp[:-1]
                        sys.stderr.flush()
                else:
                    results = dview.map_sync(constrained_foopsi_parallel, args_in)

            else:
                results = map(constrained_foopsi_parallel, args_in)

            for chunk in results:

                pars = dict()
                C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_ = chunk
                if C_ is not None:
                    Ctemp[jj_, :] = C_[None, :]
                if Sp_ is not None:
                    Stemp[jj_, :] = Sp_
                if Ytemp_ is not None:
                    Ytemp[:, jj_] = Ytemp_[:, None]
                # btemp[jj_] = cb_
                # c1temp[jj_] = c1_
                # sntemp[jj_] = sn_
                gtemp[jj_, :] = gn_.T

                if cb_ is not None:
                    bl[jo[jj_]] = cb_
                if c1_ is not None:
                    c1[jo[jj_]] = c1_
                sn[jo[jj_]] = sn_
                g[jo[jj_]] = gn_.T if kwargs['p'] > 0 else []  # gtemp[jj,:]

                pars['b'] = cb_
                pars['c1'] = c1_
                pars['neuron_sn'] = sn_
                pars['gn'] = gtemp[jj_, np.abs(gtemp[jj, :]) > 0]
                pars['neuron_id'] = jo[jj_]
                P_.append(pars)

            # YrA -= (Ctemp - C[jo, :]).T * AA[jo, :]
            YrA[::2] -= (Ctemp - C[jo, :]).T[::2].dot(AAeven[jo, :])
            YrA[1::2] -= (Ctemp - C[jo, :]).T[1::2].dot(AAodd[jo, :])

            C[jo, :] = Ctemp.copy()
            S[jo, :] = Stemp

            print str(np.sum(lo[:count + 1])) + ' out of total ' + str(nr) + ' temporal components updated'

        for ii in np.arange(nr, nr + nb):
            cc = np.maximum(YrA[:, ii] + np.atleast_2d(Cin[ii, :]).T, 0)
            # YrA -= (cc - np.atleast_2d(Cin[ii, :]).T) * AA[ii, :]
            YrA[::2] -= (cc - np.atleast_2d(Cin[ii, :]).T)[::2] * AAeven[ii, :]
            YrA[1::2] -= (cc - np.atleast_2d(Cin[ii, :]).T)[1::2] * AAodd[ii, :]
            C[ii, :] = cc.T

        if dview is not None:
            dview.results.clear()

        if scipy.linalg.norm(Cin - C, 'fro') / scipy.linalg.norm(C, 'fro') <= 1e-3:
            print "stopping: overall temporal component not changing significantly"
            break
        else:
            Cin = C.copy()

    f = C[nr:, :]
    C = C[:nr, :]
    YrA = np.array(YrA[:, :nr]).T
    P_ = sorted(P_, key=lambda k: k['neuron_id'])

    return C, f, S, bl, c1, sn, g, YrA


def interleave(Y, ds):
    tmp = np.zeros_like(Y)
    d1, d2, T = Y.shape
    for t in range(Y.shape[-1]):
        if t % 2:
            tmp[:d1 - d1 % ds, :d2 - d2 % ds, t] = Y[:d1 - d1 % ds, :d2 - d2 % ds, t]\
                .reshape(d1 / ds, ds, d2 / ds, ds).mean(1).mean(2).repeat(ds, 0).repeat(ds, 1)
            if d1 % ds > 0:
                tmp[d1 - d1 % ds:, :d2 - d2 % ds, t] = Y[d1 - d1 % ds:, :d2 - d2 % ds, t]\
                    .reshape(d1 % ds, d2 / ds, ds)\
                    .mean(0).mean(1).reshape(1, -1).repeat(d1 % ds, 0).repeat(ds, 1)
            if d2 % ds > 0:
                tmp[:d1 - d1 % ds, d2 - d2 % ds:, t] = Y[:d1 - d1 % ds, d2 - d2 % ds:, t]\
                    .reshape(d1 / ds, ds, d2 % ds)\
                    .mean(1).mean(1).reshape(-1, 1).repeat(ds, 0).repeat(d2 % ds, 1)
                if d1 % ds > 0:
                    tmp[d1 - d1 % ds:, d2 - d2 % ds:, t] = Y[d1 - d1 %
                                                             ds:, d2 - d2 % ds:, t].mean()
        else:
            k1 = (d1 - ds / 2) / ds
            k2 = (d2 - ds / 2) / ds
            n1 = ds / 2 + k1 * ds
            n2 = ds / 2 + k2 * ds
            # main area
            tmp[ds / 2:n1, ds / 2:n2, t] = Y[ds / 2:n1, ds / 2:n2, t].reshape(k1, ds, k2, ds)\
                .mean(1).mean(2).repeat(ds, 0).repeat(ds, 1)
            # sides
            tmp[n1:, ds / 2:n2, t] = Y[n1:, ds / 2:n2, t].reshape(-1, k2, ds)\
                .mean(0).mean(1).reshape(1, -1).repeat(d1 - n1, 0).repeat(ds, 1)
            tmp[ds / 2:n1, n2:, t] = Y[ds / 2:n1, n2:, t].reshape(k1, ds, -1)\
                .mean(1).mean(1).reshape(-1, 1).repeat(ds, 0).repeat(d2 - n2, 1)
            tmp[:ds / 2, ds / 2:n2, t] = Y[:ds / 2, ds / 2:n2, t].reshape(-1, k2, ds)\
                .mean(0).mean(1).reshape(1, -1).repeat(ds / 2, 0).repeat(ds, 1)
            tmp[ds / 2:n1, :ds / 2, t] = Y[ds / 2:n1, :ds / 2, t].reshape(k1, ds, -1)\
                .mean(1).mean(1).reshape(-1, 1).repeat(ds, 0).repeat(ds / 2, 1)
            # corners
            tmp[n1:, n2:, t] = Y[n1:, n2:, t].mean()
            tmp[:ds / 2, n2:, t] = Y[:ds / 2, n2:, t].mean()
            tmp[n1:, :ds / 2, t] = Y[n1:, :ds / 2, t].mean()
            tmp[:ds / 2, :ds / 2, t] = Y[:ds / 2, :ds / 2, t].mean()
    return tmp


def standardDownscale(Y, ds):
    tmp = np.zeros_like(Y)
    d1, d2, T = Y.shape
    for t in range(Y.shape[-1]):
        tmp[:d1 - d1 % ds, :d2 - d2 % ds, t] = Y[:d1 - d1 % ds, :d2 - d2 % ds, t]\
            .reshape(d1 / ds, ds, d2 / ds, ds).mean(1).mean(2).repeat(ds, 0).repeat(ds, 1)
        if d1 % ds > 0:
            tmp[d1 - d1 % ds:, :d2 - d2 % ds, t] = Y[d1 - d1 % ds:, :d2 - d2 % ds, t]\
                .reshape(d1 % ds, d2 / ds, ds)\
                .mean(0).mean(1).reshape(1, -1).repeat(d1 % ds, 0).repeat(ds, 1)
        if d2 % ds > 0:
            tmp[:d1 - d1 % ds, d2 - d2 % ds:, t] = Y[:d1 - d1 % ds, d2 - d2 % ds:, t]\
                .reshape(d1 / ds, ds, d2 % ds)\
                .mean(1).mean(1).reshape(-1, 1).repeat(ds, 0).repeat(d2 % ds, 1)
            if d1 % ds > 0:
                tmp[d1 - d1 % ds:, d2 - d2 % ds:, t] = Y[d1 - d1 % ds:, d2 - d2 % ds:, t].mean()
    return tmp


def shiftedDownscale(Y, ds):
    tmp = np.zeros_like(Y)
    d1, d2, T = Y.shape
    for t in range(Y.shape[-1]):
        k1 = (d1 - ds / 2) / ds
        k2 = (d2 - ds / 2) / ds
        n1 = ds / 2 + k1 * ds
        n2 = ds / 2 + k2 * ds
        # main area
        tmp[ds / 2:n1, ds / 2:n2, t] = Y[ds / 2:n1, ds / 2:n2, t].reshape(k1, ds, k2, ds)\
            .mean(1).mean(2).repeat(ds, 0).repeat(ds, 1)
        # sides
        tmp[n1:, ds / 2:n2, t] = Y[n1:, ds / 2:n2, t].reshape(-1, k2, ds)\
            .mean(0).mean(1).reshape(1, -1).repeat(d1 - n1, 0).repeat(ds, 1)
        tmp[ds / 2:n1, n2:, t] = Y[ds / 2:n1, n2:, t].reshape(k1, ds, -1)\
            .mean(1).mean(1).reshape(-1, 1).repeat(ds, 0).repeat(d2 - n2, 1)
        tmp[:ds / 2, ds / 2:n2, t] = Y[:ds / 2, ds / 2:n2, t].reshape(-1, k2, ds)\
            .mean(0).mean(1).reshape(1, -1).repeat(ds / 2, 0).repeat(ds, 1)
        tmp[ds / 2:n1, :ds / 2, t] = Y[ds / 2:n1, :ds / 2, t].reshape(k1, ds, -1)\
            .mean(1).mean(1).reshape(-1, 1).repeat(ds, 0).repeat(ds / 2, 1)
        # corners
        tmp[n1:, n2:, t] = Y[n1:, n2:, t].mean()
        tmp[:ds / 2, n2:, t] = Y[:ds / 2, n2:, t].mean()
        tmp[n1:, :ds / 2, t] = Y[n1:, :ds / 2, t].mean()
        tmp[:ds / 2, :ds / 2, t] = Y[:ds / 2, :ds / 2, t].mean()
    return tmp
