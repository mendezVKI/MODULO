# -*- coding: utf-8 -*-

"""
#=== FILE DESCRIPTION ======================================================
#
#   Set of python functions to write numpy array from modal decompositions to vtk file
#
#=== LICENSE ===============================================================
#
#   npyToVtk.py
#
#   Copyright 2025 Marek Belda 
#                  Martin Isoz <isozm@it.cas.cz>
#                  Tomas Hlavaty
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/gpl.html>.
#
#
#===========================================================================
"""
# ------------------- Imports -------------------
import numpy as np
import os
import vtk
from vtk.util import numpy_support
import inspect


# ------------------- Function definitions -------------------

def licenseInfo():
    """
    Prints basic licensing info
    """

    print("\n===== LICENSE INFO ===========================================")
    print("  %s  Copyright (C) 2025  Marek Belda et al. \n  This program comes with ABSOLUTELY NO WARRANTY. \n  This is free software, and you are welcome \n  to redistribute it under certain conditions. \n  For details see https://www.gnu.org/licenses/gpl.html"%os.path.basename(__file__))
    print("==============================================================\n")
    return



def prepDataForVtk(data,nDim=1,prec=6,order='',planarData=0,verbose=True):
    """
    Restores the data from modal decomposition output (flattened) 
    to their (hopefully) original form (scalar / vector field suitable for vtk).

    Parameters
    ----------
    data : 1D np.ndarray
        Vector of data to be reshaped to vtk-compatible format.
        Only accepts one-index np.ndarray (vector, not matrix) for now.
    
    nDim : int, optional
        Number of components the 'data' entry contains, Default 1 (=scalar). 
        Can take on following values:
            1: 'data' represents a spatial structure of a scalar field.
            2: 'data' represents a 2 component vector field (vtk mesh can be 3D, 
                but 'data' have only 2 directions, the 3rd is empty), 
                e.g. planar velocity field from PIV measurement.
            3: 'data' represents a 3 component vector field, 
                e.g. velocity data from CFD simulation.

    prec : int, optional
        No of decimal places to be used. Default 6.

    order : str, optional
        How to reconstruct vector field from its flattened form.
        Should be the same as the method used for flattening 
        the original field before analysis.
        Can take on following values:
            'C': C-like ordering, last index changes fastest
            'F': Fortran-like ordering, first index changes fastest
        Not required for scalar fields.

    planarData : int, optional
        Specifies the empty direction in the case of 2D vectors.
        Can take on following values:
            0: full 3D data (Default)
            1: x direction empty
            2: y direction empty
            3: z direction empty
        The empty direction gets filled with zeros to ensure compatibility 
        with vtk vector fields.
    
    verbose : bool, optional
        Provide aditional info about intermediate steps during execution. Default True.

    Returns
    -------
    Uwr : 2D np.ndarray
        Output variable of appropriate size. Contains 'data' reshaped to fit 'writeVtk' and 'appendArrToVtk' functions.
    """

    func_name = inspect.currentframe().f_code.co_name
    if verbose:
        print("Reshaping data for writing vtk ...")
    # type checks
    try:
        assert isinstance(nDim,int), "Variable 'nDim' must be integer"
        assert isinstance(prec,int), "Variable 'prec' must be integer"
        assert isinstance(order,str), "Variable 'order' must be a 1 character string"
        assert planarData in [0,1,2,3], "Incorrectly specified 'planarData', must be from [0, 1, 2, 3]. See description for details."
        assert isinstance(verbose,bool), "Variable 'verbose' must be of type 'bool'"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        print(f"ERROR IN {func_name}: No data prepaired, returning blank")
        # Exit the program immediately
        return

    if data.shape[0] % nDim != 0:
        print(f"ERROR IN {func_name}: Inconsistent data dimension")
        print(f"ERROR IN {func_name}: No data prepaired, returning blank")
        return

    length = int(data.shape[0]/nDim)
    Uwr = np.zeros((length,nDim))

    # Write data based on data dimension
    if nDim == 1:   # scalar
        if verbose:
            print(" -> Preparing scalar field ...")
        Uwr[:,0] = np.round(data,prec)
        if verbose:
            print(" -> Done.")

    elif nDim == 2 and planarData in [1,2,3]:     # 2D data, 0 padding for vtk vector compatibility
        if order == 'C':    # C-like ordering, last index changes fastest
            if verbose:
                print(" -> Reshaping data in C-like order ...")
            for k in range(nDim):
                Uwr[:,k] = np.round(data[k::nDim],prec)
        elif order == 'F':  # Fortran-like ordering, first index changes fastest
            if verbose:
                print(" -> Reshaping data in Fortran-like order ...")
            for k in range(nDim):
                Uwr[:,k] = np.round(data[k*length:(k+1)*length],prec)
        else:
            print(f"ERROR IN {func_name}: Invalid ordering specified")
            print(f"ERROR IN {func_name}: No data prepaired, returning blank")
            return

        if verbose:
            print(" -> Done.")
            print(" -> Inserting zeros for vtk vector compatibility ...")
        Uwr = np.insert(Uwr,planarData-1,values = 0,axis = 1)   # 0 padding for vtk vector compatibility
        if verbose:
            print(" -> Done.")

    elif nDim == 3:     # Full 3D vector
        if order == 'C':    # C-like ordering, last index changes fastest
            if verbose:
                print(" -> Reshaping data in C-like order ...")
            for k in range(nDim):
                Uwr[:,k] = np.round(data[k::nDim],prec)
        elif order == 'F':  # Fortran-like ordering, first index changes fastest
            if verbose:
                print(" -> Reshaping data in Fortran-like order ...")
            for k in range(nDim):
                Uwr[:,k] = np.round(data[k*length:(k+1)*length],prec)
        else:
            print(f"ERROR IN {func_name}: Invalid ordering specified")
            print(f"ERROR IN {func_name}: No data prepaired, returning blank")
            return
        if verbose:
            print(" -> Done.")

    else:
        print(f"ERROR IN {func_name}: Invalid dimension, or wrongly specified data plane normal for 2D data")
        print(f"ERROR IN {func_name}: No data prepaired, returning blank")
        return
    if verbose:
        print("Done.")
    return Uwr




def writeVtk(source, targetVTK, whatToWrite, varName, verbose=True):
    """
    Writes spatial field as vtk file for further postprocessing. If the vtk file already exists, this function overwrites it.

    Parameters
    ----------
    source : str
        Path to the source vtk or vtu file containing the desired geometry, either absolute or relative.
    
    targetVTK : str
        Output file path (output file name with a full path, either absolute or relative).
    
    whatToWrite : 2D np.ndarray
        Array containing the data that will be written to the vtk file. 
        Should have the dimension of (nCells x 1) for scalar field or (nCells x 3) for vector field.
        For seamless operation use the 'prepDataForVtk' function to obtain this array.

    varName : str
        Name to be assigned to the 'whatToWrite' array when it is written to the vtk file.

    verbose : bool, optional
        Provide aditional info about intermediate steps during execution. Default True.
    """
    func_name = inspect.currentframe().f_code.co_name
    # type checks
    try:
        assert isinstance(source,str), "Variable 'source' must be a string"
        assert isinstance(targetVTK,str), "Variable 'targetVTK' must be a string"
        assert isinstance(varName,str), "Variable 'varName' must be a string"
        assert isinstance(verbose,bool), "Variable 'verbose' must be of type 'bool'"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        print(f"ERROR IN {func_name}: No vtk files written")
        # Exit the program immediately
        return

    
    if verbose:
        print('Writing %s ...'%targetVTK)

    # Get dimensions of input array
    nDataPoints = whatToWrite.shape[0]
    nDim = whatToWrite.shape[1]

    if verbose:
        print(" -> Making output directory ...")
    # Delete target file if it already exists (make sure the writer writes into clean file)
    if os.path.exists(targetVTK):
        os.remove(targetVTK)
    # Make target directory if it does not exist
    if not os.path.exists(os.path.dirname(targetVTK)):
        os.makedirs(os.path.dirname(targetVTK))
    if verbose:
        print(" -> Done.")

    if source.endswith(".vtk"):
        # Setting up reader able to read *.vtk files (change reader if necessary)
        reader = vtk.vtkGenericDataObjectReader()
    elif source.endswith(".vtu"):
        # Setting up reader able to read *.vtu files (change reader if necessary)
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        print(f"ERROR IN {func_name}: Unsupported source file extension")
        print(f"ERROR IN {func_name}: No vtk files written")
        return

    reader.SetFileName(source)
    reader.Update()
    data = reader.GetOutput()

    # Check data dimension against no of cells
    noOfCells = data.GetNumberOfCells()
    cellType = data.GetCellType(0)
    if noOfCells != nDataPoints:
        print(f"ERROR IN {func_name}: Trying to write {nDataPoints} field entries into {noOfCells} cells")
        print(f"ERROR IN {func_name}: No vtk files written")
        return

    # Add cell field
    if verbose:
        print(" -> Adding cell field %s ..."%varName)
    vector_field = numpy_support.numpy_to_vtk(whatToWrite)
    vector_field.SetNumberOfComponents(nDim)  # 3 components for a 3D vector, 1 for scalar
    vector_field.SetName(varName)

    if nDim == 3:
        data.GetCellData().SetVectors(vector_field)
    elif nDim == 1: 
        data.GetCellData().SetScalars(vector_field)
    else:
        print(f"ERROR IN {func_name}: Invalid dimension of data, fits neither vector nor scalar fields")
        print(f"ERROR IN {func_name}: No vtk files written")
        return
    if verbose:
        print(" -> Done.")
    
    # Writer for *.vtk files
    if verbose:
        print(" -> Writing data ...")
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(targetVTK)
    writer.SetInputData(data)
    writer.Write()
    if verbose:
        print(" -> Done.")
        print(" -> Checking output file validity ...")
    # Check file validity
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(targetVTK)
    reader.Update()
    data = reader.GetOutput()

    if data.GetNumberOfCells() != noOfCells:
        print(f" -> WARNING IN {func_name}: Different no of cells in original and output file, output file may be corrupted")
    if data.GetCellType(0) != cellType:
        print(f" -> WARNING IN {func_name}: Different cell type in original and output file, output file may be corrupted")
    
    # Check for invalid cells
    for i in range(data.GetNumberOfCells()):
        cell = data.GetCell(i)
        if cell is None or cell.GetNumberOfPoints() == 0:
            print(f" -> WARNING: Missing or invalid cell at index {i}")

    if verbose:
        print(" -> Done.")
        print('%s successfully written'%os.path.basename(targetVTK))
    return



def appendArrToVtk(source, whatToWrite, varName, verbose=True):
    """
    Appends spatial field to vtk file for further postprocessing. Leaves any former content of the file intact.

    Parameters
    ----------
    source : str
        Path to the file to which the array is appended, either absolute or relative.
    
    whatToWrite : 2D np.ndarray
        Array containing the data that will be written to the vtk file. 
        Should have the dimension of (nCells x 1) for scalar field or (nCells x 3) for vector field.
        For seamless operation use the 'prepDataForVtk' function to obtain this array.

    varName : str
        Name to be assigned to the 'whatToWrite' array when it is written to the vtk file.

    verbose : bool, optional
        Provide aditional info about intermediate steps during execution. Default True.
    """

    func_name = inspect.currentframe().f_code.co_name
    # type checks
    try:
        assert isinstance(source,str), "Variable 'source' must be a string"
        assert isinstance(varName,str), "Variable 'varName' must be a string"
        assert isinstance(verbose,bool), "Variable 'verbose' must be of type 'bool'"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        print(f"ERROR IN {func_name}: No vtk files written")
        # Exit the program immediately
        return

    if verbose:
        print('Appending data to %s ...'%source)

    nDataPoints = whatToWrite.shape[0]
    nDim = whatToWrite.shape[1]

    if source.endswith(".vtk"):
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(source)
        reader.Update()
        data = reader.GetOutput()
    else:
        print(f"ERROR IN {func_name}: Unsupported file extension")
        print(f"ERROR IN {func_name}: No vtk files written")
        return

    # Check data dimension against no of cells
    noOfCells = data.GetNumberOfCells()
    cellType = data.GetCellType(0)
    if noOfCells != nDataPoints:
        print(f"ERROR IN {func_name}: Trying to write {nDataPoints} field entries into {noOfCells} cells")
        print(f"ERROR IN {func_name}: No vtk files written")
        return

    # Add cell field
    if verbose:
        print(" -> Adding cell field %s ..."%varName)
    vector_field = numpy_support.numpy_to_vtk(whatToWrite)
    vector_field.SetNumberOfComponents(nDim)  # 3 components for a 3D vector, 1 for scalar
    vector_field.SetName(varName)

    data.GetCellData().AddArray(vector_field)
    if verbose:
        print(" -> Done.")
    
    # Writer for *.vtk files
    if verbose:
        print(" -> Writing data ...")
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(source)
    writer.SetInputData(data)
    writer.Write()
    if verbose:
        print(" -> Done.")
        print(" -> Checking output file validity ...")
    # Check file validity
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(source)
    reader.Update()
    data = reader.GetOutput()

    if data.GetNumberOfCells() != noOfCells:
        print(f" -> WARNING IN {func_name}: Different no of cells in original and output file, output file may be corrupted")
    if data.GetCellType(0) != cellType:
        print(f" -> WARNING IN {func_name}: Different cell type in original and output file, output file may be corrupted")
    
    # Check for invalid cells
    for i in range(data.GetNumberOfCells()):
        cell = data.GetCell(i)
        if cell is None or cell.GetNumberOfPoints() == 0:
            print(f" -> WARNING IN {func_name}: Missing or invalid cell at index {i}")

    if verbose:
        print(" -> Done.")
        print('Successfully appended data to %s'%os.path.basename(source))
    return




def writeAll(mat,nameList,targetVTKList,source,nDim=1,prec=6,order='',planarData=0,saveAsOneFile=False,verbose=False):
    """
    Wrapper for the 'prepDataForVtk', 'writeVtk' and 'appendArrToVtk' functions. 
    Automates most of the actions previously performed by the user.

    Parameters
    ----------
    mat : 2D np.ndarray
        Matrix to be saved to vtk (columnwise). Each column is one data vector.

    nameList : list of str
        Names to be assigned to columns of 'mat' when writing to vtk.
        
    targetVTKList : list of str
        List of paths for output files (each entry is output file name with a full path, either absolute or relative).
    
    source : str
        Path to the source vtk or vtu file, either absolute or relative.

    nDim : int, optional
        Number of dimensions the 'mat' data contain. 
        For further info see 'prepDataForVtk'.
        Default 1.

    prec : int, optional
        No of decimal places to be used. Default 6.

    order : str, optional
        How to reconstruct vector field from its flattened form.
        See 'prepDataForVtk' for more info

    planarData : int, optional
        Specifies the empty direction in the case of 2D vectors.
        See 'prepDataForVtk' for more info

    saveAsOneFile : bool, optional
        Whether to compress data into one vtk. Default False.

    verbose : bool, optional
        Provide aditional info about intermediate steps during execution. Default False.
        Gets passed to any called functions and overrides the default values there. 
    """
    func_name = inspect.currentframe().f_code.co_name
    print("Splitting data and saving to vtk ...")

    # type checks
    try:
        assert isinstance(nameList,list), "Variable 'nameList' must be a list"
        assert isinstance(targetVTKList,list), "Variable 'targetVTKList' must be a list"
        assert isinstance(saveAsOneFile,bool), "Variable 'saveAsOneFile' must be of type 'bool'"
        assert isinstance(verbose,bool), "Variable 'verbose' must be of type 'bool'"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        print(f"ERROR IN {func_name}: No vtk files written")
        # Exit the program immediately
        return

    noOfFiles = mat.shape[1]

    if not saveAsOneFile:
        if not (len(nameList) == noOfFiles and len(targetVTKList) == noOfFiles):    # Check no of entries
            print(f"ERROR IN {func_name}: Number of entries in 'nameList' or 'targetVTKList' does not match number of files to be saved.")
            print(f"ERROR IN {func_name}: No vtk files written")
            return 
        
        for k in range(noOfFiles):      # Write each column of 'mat' into its own file
            print("Prepairing %s and writing it to %s"%(nameList[k],targetVTKList[k]))
            vecExp = prepDataForVtk(mat[:,k],nDim,prec,order,planarData,verbose=verbose)
            writeVtk(source,targetVTKList[k],vecExp,nameList[k],verbose=verbose)

    else:
        if not (len(nameList) == noOfFiles and len(targetVTKList) == 1):    # Check no of entries
            print(f"ERROR IN {func_name}: Number of entries in 'nameList' does not match no of arrays to be saved or 'targetVTKList' does not have one entry.")
            print(f"ERROR IN {func_name}: No vtk files written")
            return 

        for k in range(noOfFiles):      # Write each column of 'mat' into its own array
            print("Prepairing %s and writing it to %s"%(nameList[k],targetVTKList[0]))
            vecExp = prepDataForVtk(mat[:,k],nDim,prec,order,planarData,verbose=verbose)
            if k == 0:
                writeVtk(source,targetVTKList[0],vecExp,nameList[k],verbose=verbose)
            else:
                appendArrToVtk(targetVTKList[0], vecExp, nameList[k],verbose=verbose)

    print("All data saved.")
    return
    