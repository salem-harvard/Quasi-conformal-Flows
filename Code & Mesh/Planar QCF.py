# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 23 2023

Author: Salem Mosleh
"""

import jax.numpy as np
from jax.numpy import cos, sin

import jax.numpy.linalg as la
import jax.random as npr
key = npr.PRNGKey(0)

import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)

from jax import vmap, jit

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
import jaxopt

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
from pathlib import WindowsPath
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join

cwd = Path(os.getcwd())

resultsDir = cwd  / "Run Results" 

meshDir = cwd / "Mesh Data"
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------




runNames = ["disc_shear", "hawk_moth", "buckeye_butterfly"]
paramterTypes = ["conformal", "elastic", "uniform", "conformal-forced"] #


spatialDim = 2; numTimeSteps = 30;




#===========================================================================================================
#   Run optimization for all time steps
#===========================================================================================================
def allTimeSteps(runIndx=0, paramIndex=0, startingInd=0):
    '''
    '''
    
    
    runName = runNames[runIndx]
    paramterType = paramterTypes[paramIndex]
    if paramterType=="conformal":
        a=0; b=1; c=0;d=0;e=0;  regC=100000;
    elif paramterType=="conformal-forced":
        a=0; b=10; c=0;d=0;e=0;  regC=100;
    elif paramterType == "elastic":
        a=1; b=1;  c=0;d=0;e=0; regC=100000; 
    elif paramterType=="uniform":
        a=0; b=0; c=0.1;d=0.1;e=0.0; regC=100000; 
    else: 
        raise TypeError("no paramter choise given")
    
    

    
    
    if not os.path.exists(resultsDir / runName):
        os.makedirs(resultsDir / runName)
    
    
    allMaxDeformation = 0
    
    print("working on data for: " + runName)
    
    for stepNum in range(startingInd, numTimeSteps):
        
        mesh_name =   paramterType + "_time" + str(stepNum)
        
        print("working on: " + mesh_name + "_" + runName)
        
        newVerts, initStrains , finalStrains, maxDeformation = stepInTime(stepNum, runIndx,
                             paramIndex, a, b, c, d, e, regC)
    
        
        

        
        print("saving results for step number: " + str(stepNum))

        onp.savetxt(resultsDir / runName / ("strains_" + mesh_name + ".csv") , finalStrains, delimiter=',')            
        onp.savetxt(resultsDir / runName / ("new_verts_" + mesh_name + ".csv") , newVerts, delimiter=',')
        
        
        onp.savetxt(resultsDir / runName / ("init_strains_" + mesh_name + ".csv") , initStrains, delimiter=',')        

        allMaxDeformation = onp.max([allMaxDeformation, maxDeformation])
  

        
    
    onp.savetxt(resultsDir / runName / (paramterType + "_MaxDilations.csv") , [allMaxDeformation], delimiter=',') 
    
    return 
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================




#===========================================================================================================
#   Run optimization for a single time step
#===========================================================================================================
def stepInTime(stepNum, runIndx, paramIndex, a, b, c, d, e, regC):
    '''
    '''
    

    
    runName = runNames[runIndx]
    paramterType = paramterTypes[paramIndex]
    
    loadDir = meshDir / runName / ("time" + str(stepNum))
    loadDir0 = meshDir / runName / ("time" + str(0)) 
    
    
    
    
    vertices0 = onp.loadtxt(loadDir / "vertices.csv", delimiter=",", dtype=np.float64)
    
    
    vertices1 = onp.loadtxt(loadDir / "vertices_deformed.csv", delimiter=",", dtype=np.float64) #"normal_deformed_intersection.csv"
    
    faces = onp.loadtxt(loadDir / "faces.csv", delimiter=",", dtype=np.int32)
    numFaces = len(faces)
    #faceNeighborList = onp.loadtxt(loadDir / "facesNeibs.csv", delimiter=",", dtype=np.int32)

    
    boundaryInds = onp.loadtxt(loadDir / "boundary_indices.csv", delimiter=",", dtype=np.int32)

    
    boundaryNormals = onp.loadtxt(loadDir / "normal.csv", delimiter=",", dtype=np.float64)
        
    boundaryLengths = onp.loadtxt(loadDir / "boundaryLengths.csv", delimiter=",", dtype=np.float64)
    
    gradCostMat = onp.loadtxt(loadDir / "gradCostMat.csv", delimiter=",", dtype=np.float64)
    
    areas = onp.loadtxt(loadDir / "faceAreas.csv", delimiter=",", dtype=np.float64)    
    areaMat = np.diag(areas)
    

    
    initialArea = np.sum(onp.loadtxt(loadDir0 / "faceAreas.csv", delimiter=",", dtype=np.float64))
    
    
    duration = 30
    
    
    
    #when penalizing variations in time
    if(stepNum==0):
        E = 0
        previousStrains = onp.zeros((numFaces, spatialDim, spatialDim))
    else:
        E = e
        prevStepNum = onp.max([stepNum - 1, 0])
        file_name =   "strains_" + paramterType + "_time" + str(prevStepNum) + ".csv"
        previousStrains = onp.loadtxt(resultsDir / runName / file_name ,  delimiter=',')            
        previousStrains = onp.reshape(previousStrains, (numFaces, spatialDim, spatialDim)) 
    
    
    
    displacements = vertices1 - vertices0
    
    initBoundaryDisps = displacements[boundaryInds]
    
    #alpha = 4; beta= 4;
    #params = np.row_stack((displacements, [alpha, beta]))
    
    # return cost(displacements, startingVerts=vertices0, tris=faces, areas=areas,
    #                   boundaryInds=boundaryInds, normals=boundaryNormals, gradCostMat=gradCostMat,
    #                   a=a, b=b, dGrad=dGrad,  sGrad=sGrad, regC=regC)
    

    solver  = jaxopt.LBFGS(fun=cost)
    res = solver.run(displacements, startingVerts=vertices0, initBoundaryDisps=initBoundaryDisps, 
                     faces=faces, areaMat=areaMat, boundaryInds=boundaryInds, 
                     normals=boundaryNormals, gradCostMat=gradCostMat, boundaryLengths=boundaryLengths, 
                      prevStrains=previousStrains, initialArea=initialArea, duration=duration,
                      a=a, b=b, c=c,  d=d, e=E, regC=regC)
    
    
    

    newDisps, s = res

    #nondimensionalizing the strain rate by multiplying by time
    initStrains = duration*getDeformations(vertices0, displacements, faces)
    
    finalStrains = duration*getDeformations(vertices0, newDisps, faces)
    
    newVerts = vertices0 + newDisps
    
    maxDeformation = getMaxDilationAndShear(finalStrains)
    
    return newVerts, onp.reshape(initStrains, (numFaces, spatialDim**2)), onp.reshape(finalStrains, (numFaces, spatialDim**2)), maxDeformation


#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================





#===========================================================================================================
#   Add the elastic cost for all simplices
#===========================================================================================================
@jit
def cost(disps, startingVerts, initBoundaryDisps, faces, areaMat,
         boundaryInds, normals, gradCostMat, boundaryLengths, prevStrains,
          initialArea, duration, a, b, c, d, e, regC) : 
    '''
    '''
    
    strains = getDeformations(startingVerts, disps, faces)
    
    cost = 0

    dilationMat = (duration/initialArea) * (a - b/2)*areaMat + duration * (c - d/2)*gradCostMat
    shearMat = (duration/initialArea) * b * areaMat + duration * d * gradCostMat
    
    cost += np.einsum('kii, kl, ljj', strains , dilationMat, strains)
    cost += np.einsum('kij, kl, lij', strains , shearMat, strains)


    #penalize changes over time of the deformations
    strainChange = strains - prevStrains
    changeParam = e*(duration**3)/initialArea
    cost += changeParam*np.einsum('kii, kl, ljj', strainChange , areaMat, strainChange)
    cost += changeParam*np.einsum('kij, kl, lij', strainChange , areaMat, strainChange)

    #enforce boundary displacements are tangent to the shape.   
    cost += (duration/initialArea**1.5)*regC*((
        np.einsum('ij, i,ij -> i', disps[boundaryInds] - initBoundaryDisps, boundaryLengths, normals)**2).sum())
      
    
    return cost
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================





#===========================================================================================================
# Calculate the shear and dilation given the initial vertices and displacements
#===========================================================================================================   
@jit
def getDeformations(vertices, displacements, faces):

    
    v_mapped = vmap(cost_per_simplex2, (0, None, None))
    
    #dilations, shears, traceless = v_mapped(faces, vertices, displacements) 
        
    return  v_mapped(faces, vertices, displacements) #dilations, shears, traceless
#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================== 





#===========================================================================================================
#   Calculate the elastic cost per simplex using face index
#===========================================================================================================
@jit
def cost_per_simplex2(face, allVerts, allDisps):

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
    
    init_positions = allVerts[face]
    displacements = allDisps[face]
    
    #strain = calculateStrains(init_positions, displacements)
    
    
    return calculateStrains(init_positions, displacements)#getDilationAndShear(strain)
    


#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================   





#===========================================================================================================
#   Calculate the elastic cost per simplex
#===========================================================================================================
@jit
def calculateStrains(init_positions, displacements):

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
    
    #projection = np.dot(basis,dualBasis)
    
    strain = np.dot(deltaV, dualBasis)
    
    return (strain + strain.transpose())*0.5
    
    #return area *  (a * np.trace(strain)**2 + b * np.trace(np.dot(strain,strain)))/ 8


#===========================================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#===========================================================================================================    




# #===========================================================================================================
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# #=========================================================================================================== 
# @jit
# def getDilationAndShear(strain):
    
#     dilation = np.trace(strain)/spatialDim
    
#     traceless = strain - dilation*np.identity(spatialDim)
    
#     shear = np.sqrt(np.trace(np.dot(traceless,traceless)))/spatialDim
    
#     return dilation, shear, traceless
# #===========================================================================================================
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# #=========================================================================================================== 


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
    


# for i in range(len(runNames)):
#     for j in range(len(paramterTypes)):
#         allTimeSteps(runIndx=i, paramIndex=j)
          
# for j in range(len(paramterTypes)):
#     allTimeSteps(runIndx=2, paramIndex=j)

for i in range(1, len(runNames)):
    allTimeSteps(runIndx=i, paramIndex=2)
 
    
# allTimeSteps(runIndx=1, paramIndex=1, startingInd=6) 



    