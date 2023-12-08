# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 23 2023

Author: Salem Mosleh
"""


import jax.numpy as np
import numpy.random as onpr
from jax.numpy import cos, sin

import jax.numpy.linalg as la
import jax.random as npr
key = npr.PRNGKey(0)

import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)

from jax import vmap, jit


import jaxopt

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
from pathlib import Path
import os
# from os import listdir
# from os.path import isfile, join

cwd = Path(os.getcwd())

resultsDir = cwd  / "Run Results" 

meshDir = cwd / "Mesh Data"
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------




runNames = ["Wigly_Cylinder", "leaves"]
paramterTypes = ["ricci", "mean", "all", "none"]

spatialDim = 3; numTimeSteps = 30




#===========================================================================================================
#   Run optimization for all time steps
#===========================================================================================================
def allTimeSteps(runIndx=0, paramIndex=0, startInd=0):
    '''
    allTimeSteps(runIndx=0, a=10, b=0,  dGrad=1, sGrad=1, regC=1000, regG=1)
    
    calculates the optimal registration by minimizing the given cost. For each time step
    which is indexed by runIndx. 
    
    a: dilation cost coefficient. which also may include growth costs 
    
    b: shear cost coefficient.
    
    dGrad: dilation gradient cost coefficient.
    
    sGrad: shear gradient cost coefficient.
    
    regC: enforces the constraint that fixed the normal displacment. Only tangent displacements
    are free to move, corresponding to different registrations (diffeomorphisms).
    
    regG: coefficient penalizing large growth paramter values.
    
    '''
    
    runName = runNames[runIndx]
    paramterType = paramterTypes[paramIndex]
    
    if paramterType=="ricci":
        a1=0.0; b1=1.0; c1 = 0.0; d1 = 0.0; a2=1; b2=1; e2=0.0; regG = 0.01; regC=100000;
        gCoeffs = np.array([0,0,1])
    elif paramterType=="mean":
        a1=0.0; b1=0.0; c1 = 0.0; d1 = 0.0; a2=1; b2=1; e2=0.0; regG = 0.01; regC=100000;
        gCoeffs = np.array([0,1,0])
    elif paramterType=="all":
        a1=0.0; b1=0.0;c1 = 0.0; d1 = 0.0; a2=1; b2=1; e2=0.0; regG = 0.01; regC=100000;
        gCoeffs = np.array([1,1,1]) 
    elif paramterType=="none":
        a1=0.0; b1=0.0;c1 = 0.0; d1 = 0.0; a2=1; b2=1; e2=0.0; regG = 0.01; regC=100000;
        gCoeffs = np.array([0,0,0])     
    else: 
        raise TypeError("no paramter choise given")
    
    
    
    if not os.path.exists(resultsDir / runName):
        os.makedirs(resultsDir / runName)
        
    
    print("working on data for: " + runName)
    
    allMaxDeformation1 = 0
    allMaxDeformation2 = 0
    
    for stepNum in range(startInd,numTimeSteps):
        
        mesh_name =   paramterType + "_time" + str(stepNum)
        
        print("working on: " + mesh_name + "_" + runName)
        
        
        newVerts, initStrains1 , finalStrains1, initStrains2 , finalStrains2, newGParams, maxDeformation1, maxDeformation2 = stepInTime(
            stepNum, runIndx, paramIndex, a1, b1, c1, d1, a2, b2, e2, regC, regG, gCoeffs)
    
        

        
        print("saving results for step number: " + str(stepNum))
        
        onp.savetxt(resultsDir / runName / ("init_strains_" + mesh_name + ".csv") , initStrains1, delimiter=',')   
        onp.savetxt(resultsDir / runName / ("strains_" + mesh_name + ".csv") , finalStrains1, delimiter=',')
        
        onp.savetxt(resultsDir / runName / ("growth_init_strains_" + mesh_name + ".csv") , initStrains2, delimiter=',')   
        onp.savetxt(resultsDir / runName / ("growth_strains_" + mesh_name + ".csv") , finalStrains2, delimiter=',')

            
        onp.savetxt(resultsDir / runName / ("new_verts_" + mesh_name + ".csv") , newVerts, delimiter=',')
        
        allMaxDeformation1 = onp.max([allMaxDeformation1, maxDeformation1])
        allMaxDeformation2 = onp.max([allMaxDeformation2, maxDeformation2])
        
        

        onp.savetxt(resultsDir  / runName / ("growthParameters_" + mesh_name + ".csv") , newGParams, delimiter=',') 
        
        
        
        
    onp.savetxt(resultsDir / runName / (paramterType + "_MaxDilations.csv") , [allMaxDeformation1], delimiter=',')  
    onp.savetxt(resultsDir / runName / (paramterType + "_MaxDilationsGrowth.csv") , [allMaxDeformation2], delimiter=',')  
  
    
  
    return 
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================




#===========================================================================================================
#   Run optimization for a single time step
#===========================================================================================================
def stepInTime(stepNum, runIndx, paramIndx, a1, b1, c1, d1, a2, b2, e2, regC, regG, gCoeffs):
    '''
    '''
    
    runName = runNames[runIndx]
    paramterType = paramterTypes[paramIndx]
    
    loadDir = meshDir / runName / ("time" + str(stepNum))    
    
    vertices0 = onp.loadtxt(loadDir / "vertices.csv", delimiter=",", dtype=np.float64)
    
    vertices1 = onp.loadtxt(loadDir / "vertices_deformed.csv", delimiter=",", dtype=np.float64) #"normal_deformed_intersection.csv"
    
    faces = onp.loadtxt(loadDir / "faces.csv", delimiter=",", dtype=np.int32)
    numFaces = len(faces)
    
    normalDistances = onp.loadtxt(loadDir / "distance.csv", delimiter=",", dtype=np.float64)

    
    normals = onp.loadtxt(loadDir / "normal.csv", delimiter=",", dtype=np.float64)
    
    
    boundaryInds = onp.loadtxt(loadDir / "boundary_indices.csv", delimiter=",", dtype=np.int32)
    
    #normal to the boundary and parallel to the surface
    boundaryNormals = onp.loadtxt(loadDir / "boundary_normal.csv", delimiter=",", dtype=np.float64)
    
    boundaryLengths = onp.loadtxt(loadDir / "boundaryLengths.csv", delimiter=",", dtype=np.float64)
    
    gradCostMat = onp.loadtxt(loadDir / "gradCostMat.csv", delimiter=",", dtype=np.float64)
   
    areas = onp.loadtxt(loadDir / "faceAreas.csv", delimiter=",", dtype=np.float64)
    areaMat = np.diag(areas)
    
    loadDir0 = meshDir / runName / ("time" + str(0))   
    initialArea = np.sum(onp.loadtxt(loadDir0 / "faceAreas.csv", delimiter=",", dtype=np.float64))
    duration = 30
    
    
    vertexAreas = onp.loadtxt(loadDir / "vertexAreas.csv", delimiter=",", dtype=np.float64)
    
    
    gaussCurvatures = onp.loadtxt(loadDir / "Cgaussian.csv", delimiter=",", dtype=np.float64)
    meanCurvatures = onp.loadtxt(loadDir / "Cmean.csv", delimiter=",", dtype=np.float64)
    
    curvatureTensors = onp.loadtxt(loadDir / "CurvatureTensors.csv", delimiter=",", dtype=np.float64)
    curvatureTensors = onp.reshape(curvatureTensors, (numFaces, spatialDim, spatialDim)  )
    
    
    #when penalizing variations in time
    if(stepNum==0):
        E2 = 0
        previousGParams = np.zeros(3)
    else:
        E2 = e2
        prevStepNum = onp.max([stepNum - 1, 0])
        file_name =   "growthParameters_" + paramterType + "_time" + str(prevStepNum) + ".csv"
        previousGParams = onp.loadtxt(resultsDir  / runName / file_name , delimiter=',') 
  
    

    
    
    displacements = vertices1 - vertices0
    
    
    initBoundaryDisps = displacements[boundaryInds]
    
    
    dt = 1
    alpha, beta, gamma = dt*onpr.uniform(-1, 1, 3); 
    
    gParams = gCoeffs*np.array( [alpha, beta, gamma])
    params = np.row_stack((displacements, gParams))
    
    # return cost(displacements, startingVerts=vertices0, tris=faces, areas=areas,triNeiborList=faceNeighborList,
    #                   boundaryInds=boundaryInds, normals=boundaryNormals, gradCostMat=gradCostMat,
    #                   a=a, b=b, dGrad=dGrad,  sGrad=sGrad, regC=regC)
    

    solver  = jaxopt.LBFGS(fun=cost)
    res = solver.run(params, startingVerts=vertices0,  initDisps=displacements,
                     faces=faces, areaMat=areaMat, normals=normals, gradCostMat=gradCostMat,
                     initBoundaryDisps=initBoundaryDisps,  boundaryInds=boundaryInds, boundaryNormals=boundaryNormals,
                     allBij=curvatureTensors, allGaussK = gaussCurvatures, allMeanK = meanCurvatures,
                     PrevParams=previousGParams, gCoeffs=gCoeffs, vertexAreas=vertexAreas,
                      boundaryLengths=boundaryLengths, initialArea=initialArea, duration=duration,
                      a1=a1, b1=b1, c1=c1, d1=d1, a2=a2, b2=b2, e2=E2, regC=regC, regG=regG)
    
    
    
 
    

    newParams, s = res

    newDisps = newParams[:-1]
    
    
    gFactor = np.array([1, initialArea, initialArea])/duration
    newGParams = gCoeffs*newParams[-1]
    

    initStrains1, initStrains2 = getDeformations(vertices0, displacements, faces,
                                                                    gParams*gFactor, curvatureTensors, gaussCurvatures, meanCurvatures)
    
    finalStrains1, finalStrains2 = getDeformations(vertices0, newDisps, faces,
                                                                    newGParams*gFactor, curvatureTensors, gaussCurvatures, meanCurvatures)
    
    newVerts = vertices0 + newDisps
    
    maxDeformation1 = getMaxDilationAndShear(finalStrains1)
    maxDeformation2 = getMaxDilationAndShear(finalStrains2)
    
    
    return newVerts, onp.reshape(duration*initStrains1, (numFaces, spatialDim*spatialDim)), onp.reshape(
        duration*finalStrains1, (numFaces, spatialDim*spatialDim)  ), onp.reshape(duration*initStrains2, (numFaces, spatialDim*spatialDim)
         ), onp.reshape(duration*finalStrains2, (numFaces, spatialDim*spatialDim)  ),  newGParams, maxDeformation1, maxDeformation2



#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================



#===========================================================================================================
#   Add the elastic cost for all simplices
#===========================================================================================================
@jit
def cost(params, startingVerts, initDisps, faces, areaMat,  normals, gradCostMat, 
         initBoundaryDisps,  boundaryInds, boundaryNormals, allBij, allGaussK, allMeanK,
          PrevParams, gCoeffs, vertexAreas, boundaryLengths, initialArea, duration, 
          a1, b1, c1, d1, a2, b2, e2, regC, regG) :
    '''
    '''
    
    disps = params[:-1]

    gParams = gCoeffs*params[-1]*np.array([1, initialArea, initialArea])/duration
    
    
    strain1, strain2 = getDeformations(startingVerts, disps, faces, gParams, allBij, allGaussK, allMeanK)
    
    cost = 0

    dilationMat1 = (duration/initialArea) * (a1 - b1/2)*areaMat + duration * (c1 - d1/2)*gradCostMat
    shearMat1 = (duration/initialArea) * b1 * areaMat + duration * d1 * gradCostMat
    
    cost += np.einsum('kii, kl, ljj', strain1 , dilationMat1, strain1)
    cost += np.einsum('kij, kl, lij', strain1 , shearMat1, strain1)
    
    
    dilationMat2 = (duration/initialArea) * (a2 - b2/2)*areaMat 
    shearMat2 = (duration/initialArea) * b2 * areaMat 
    
    cost += np.einsum('kii, kl, ljj', strain2 , dilationMat2, strain2)
    cost += np.einsum('kij, kl, lij', strain2 , shearMat2, strain2)
    
    
    # penalize changes over time of the growth parameters
    cost +=  duration*e2*np.sum((PrevParams - gParams)**2)
    
    #regularize the growth paramters so that they want to be zero    
    cost += (1/duration)*regG*np.sum(gParams**2)
    
    #enforce the normal displacement which is approximately independent of registration
    #cost += (duration/initialArea**2) * regC * ( vertexAreas*(np.einsum('ij, ij -> i', disps , normals) - initNormalDisps)**2 ).sum()
    cost += (duration/initialArea**2) * regC * ((
        np.einsum('ij, i,ij -> i', disps - initDisps, vertexAreas, normals)**2).sum())
    
    #enforce boundary displacements are tangent to the shape.   
    cost += 0*(duration/initialArea**1.5)*regC*((
        np.einsum('ij, i,ij -> i', disps[boundaryInds] - initBoundaryDisps, boundaryLengths, boundaryNormals)**2).sum())

    
    return cost
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================

 


#===========================================================================================================
# Calculate the shear and dilation given the initial verts and displacements
#===========================================================================================================   
@jit
def getDeformations(vertices, displacements, faces, gParams, allBij, allGaussK, allMeanK):

    
    v_mapped = vmap(cost_per_simplex, (0, None, None, None, 0, 0, 0))
    
    return v_mapped(faces, vertices, displacements, gParams, allBij, allGaussK, allMeanK) 
        
    #return  dilations1, shears1, traceless1, dilations1, shears1
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 




#===========================================================================================================
#   Calculate the elastic cost per simplex using face index
#===========================================================================================================
@jit
def cost_per_simplex(face, allVerts, allDisps, gParams, bij, gaussK, meanK):

    '''
    This will give the contribution to the cost (or metric) from each simplex
    
    init_positions: 
    List with shape (3,3) of the positions of the vertices of the triangle on 
    the original plane.
    
    displacements: 
    List with shape (3,3) of the changes in positions of the vertices of the triangle on 
    the deformed surface
    
    a: is the coefficient of the area preserving changes. 
    b: is the coefficient of area changes
    '''
    

    # bij = allBij[faceInd]
    # gaussK = allGaussK[faceInd]
    # meanK = allMeanK[faceInd]
    
    init_positions = allVerts[face]
    displacements = allDisps[face]
    
    
    strain1, strain2 =  calculateStrains(init_positions, displacements, gParams, bij, gaussK, meanK)
    

    return strain1, strain2

#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================   



#===========================================================================================================
#   Calculate the elastic cost per simplex
#===========================================================================================================
@jit
def calculateStrains(init_positions, displacements, gParams, bij, gaussK, meanK):

    '''
    This will give the contribution to the cost (or metric) from each simplex
    
    init_positions: 
    List with shape (3,3) of the positions of the vertices of the triangle on 
    the original plane.
    
    displacements: 
    List with shape (3,3) of the changes in positions of the vertices of the triangle on 
    the deformed surface
    
    a: is the coefficient of the area preserving changes. 
    b: is the coefficient of area changes
    '''
    
    
    vb1 = init_positions[1] - init_positions[0]
    vb2 = init_positions[2] - init_positions[0]
    
    basis = np.array([vb1, vb2]).transpose()
    dualBasis = la.pinv(basis)
    
    
    deltaV1 = displacements[1] - displacements[0]
    deltaV2 = displacements[2] - displacements[0]
    
    deltaV = np.array([deltaV1, deltaV2]).transpose()
    
    #print("deltaV shape:" + str(deltaV.shape))
    #print("dualBasis shape:" + str(dualBasis.shape))
    
    projection = np.dot(basis, dualBasis)
    
    
    strain = np.dot(projection, np.dot(deltaV, dualBasis))
    
    strain1 = 0.5*(strain + strain.transpose())
    
    growthLaw = gParams[0]*projection + gParams[1]*meanK*bij + gParams[2]*gaussK*projection
    
    strain2 = strain1 - growthLaw
    
    
    
    return strain1, strain2
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================    
    



#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 
def getMaxDilationAndShear(strain):
    
    #finding the maximum dilation
    dilations = np.abs(np.einsum('kii -> k', strain)); maxDilation = onp.max(dilations)
    
    
    shears = np.sqrt(2*np.einsum('kij, kij -> k', strain, strain) - dilations**2)
    maxShear = onp.max(shears)
    
    maxDeformation = onp.max([maxDilation, maxShear])
    
    return maxDeformation
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 
    
 
    
#run the code looping over cases of interest
#-----------------------------------------------------------------------------------------------------------     
#-----------------------------------------------------------------------------------------------------------
# for i in range(len(runNames)):
#     for j in range(len(paramterTypes)):
#         allTimeSteps(runIndx=i, paramIndex=j) 





allTimeSteps(runIndx=0, paramIndex=2,startInd=20)
allTimeSteps(runIndx=0, paramIndex=3)



for j in range(len(paramterTypes)):
    allTimeSteps(runIndx=1, paramIndex=j) 

# for i in range(len(runNames)):
#     allTimeSteps(runIndx=i, paramIndex=1) 
  
 

#-----------------------------------------------------------------------------------------------------------   
#-----------------------------------------------------------------------------------------------------------       
 
    
 
