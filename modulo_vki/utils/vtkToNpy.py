# -*- coding: utf-8 -*-

"""
#=== FILE DESCRIPTION ======================================================
#
#   Set of python functions to write numpy array from modal decompositions to vtk file
#
#=== LICENSE ===============================================================
#
#   vtkToNpy.py
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
import re
import vtk.util.numpy_support as vtk_np
import pyvista as pv
import sys
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




def atof(text):
    """
    Sorting using regex
    """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval




def natural_keys(text):
    '''
    Regex keys for sorting
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]




def readAll(dirPath,fieldName,sortBy='',ignoreDirs=[],fieldType='scalar',order='',saveMatrix=False,savePath='./all.npy',verbose=False,skipFilePrint=1):
    """
    Extracts all vtk / vtu files in parent directory and its subdirs and saves the data in matrix form. Skips all dirs it is told to skip.

    Parameters
    ----------
    dirPath : str
        Path to the parent directory, absolute or relative.
    
    fieldName : str
        Name of the field to be extracted.

    sortBy : str, optional
        Sorting method for data. If not specified, no sorting is applied. Can take on following values:
            'dirs': assumes one readable (i.e. with .vtk / .vtu extension) file in each subdir named the same
                    across all subdirs, sorts by subdir names, e.g. output of OpenFOAM foamToVTK utility
            'files': sorts by file names, useful when there are no subdirs, only files in 'dirPath'
            'dirsAndFiles': sorts first by subdir names, then by file names, 
                            most general, useful when there are multiple files in each subdir

    ignoreDirs : list of str, optional
        List of subdirectories to skip, absolute path required !

    fieldType : str, optional
        What type of field to expect. Can take on following values:
            'scalar' = scalar field (pressure, density, ...)
            'vector' = 3D vector field (velocity, vorticity, ...)
        Default 'scalar'.

    order : str, optional
        How to flatten vector field from its vtk representation.
        Can take on following values:
            'C': C-like ordering, last index changes fastest (flattened field looks like [u1_x, u1_y, u1_z, u2_x, ...])
            'F': Fortran-like ordering, first index changes fastest (flattened field looks like [u1_x, u2_x, ..., u1_y, u2_y, ...])
        Not required for scalar fields.

    saveMatrix : bool, optional
        Whether to save the matrix to a specified place and return it (True) or only return the matrix (False).
        Default False.

    savePath : str, optional
        Where to save the resulting matrix. Relevant only when saveMatrix == True. Default './all.npy'.

    verbose : bool, optional
        Provide aditional info about intermediate steps during execution. Default False.

    skipFilePrint : int >= 1, optional
        Defines, how often to report file that is being read (e.g. 10: report every 10th file). 
        Default 1 (report every file).
    
    Returns
    -------
    out : 2D np.ndarray
        Output variable of size (nDim*nCells x nFiles). 
        In each column contains all data from one vtk file. Has as many columns as there were vtk files to read.
    """
 
    print("Extracting %s field %s from vtk / vtu files ..."%(fieldType,fieldName))
    func_name = inspect.currentframe().f_code.co_name
    # type checks
    try:
        assert (isinstance(dirPath,str) and os.path.exists(dirPath)), "Variable 'dirPath' is not string or is not a valid path"
        assert isinstance(fieldName,str), "Variable 'fieldName' must be a string"
        assert (isinstance(ignoreDirs,list) and all(isinstance(item, str) for item in ignoreDirs)), "Variable 'ignoreDirs' must be a list of strings"
        assert isinstance(order,str), "Variable 'order' must be a 1 character string"
        assert isinstance(saveMatrix,bool), "Variable 'saveMatrix' must be of type 'bool'"
        assert isinstance(savePath,str), "Variable 'savePath' must be a string"
        assert isinstance(verbose,bool), "Variable 'verbose' must be of type 'bool'"
        assert isinstance(skipFilePrint,int), "Variable 'skipFilePrint' must be of type 'int' and >=1"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        # Exit the program immediately
        sys.exit(1)

    # File reporting    
    if skipFilePrint == 1:
        print("Reporting every file")
    elif skipFilePrint == 2:
        print("Reporting every %dnd file"%skipFilePrint)
    elif skipFilePrint == 3:
        print("Reporting every %drd file"%skipFilePrint)
    elif skipFilePrint > 3:
        print("Reporting every %dth file"%skipFilePrint)
    else:
        print(f"WARNING IN {func_name}: Invalid skipFilePrint option specified, reverting to default")
        skipFilePrint=1


    ignoreDirs = [os.path.abspath(d) for d in ignoreDirs]

    # Going through all subdirs and files
    i = 0
    for root, dirs, fileList in os.walk(dirPath,topdown=True): 
        # Remove ignored dirs from traversal
        dirs[:] = [d for d in dirs if not any(ig in os.path.abspath(os.path.join(root, d)) for ig in ignoreDirs)]
        if sortBy == 'dirs':
            if verbose:
                print(" -> Sorting by subdir names ...")
            dirs.sort(key=natural_keys)     # sort dirs to appear in correct order
            if verbose:
                print(" -> Done.")
        elif sortBy == 'files':
            if verbose:
                print(" -> Sorting by file names ...")
            fileList.sort(key=natural_keys)  # sort files to appear in correct order
            if verbose:
                print(" -> Done.")
        elif sortBy == 'dirsAndFiles':
            if verbose:
                print(" -> Sorting by subdir names, then by file names ...")
            dirs.sort(key=natural_keys)     # sort dirs to appear in correct order
            fileList.sort(key=natural_keys)  # sort files to appear in correct order
            if verbose:
                print(" -> Done.")
        else:
            print(f"WARNING IN {func_name}: Invalid sorting spec, proceeding without sorting")

        # Reading all files
        for file in fileList:
            if file.endswith('.vtu') or file.endswith('.vtk'):
                fileToRead = os.path.join(root, file)
                if i % skipFilePrint == 0:
                    print("Processing file: " + str(fileToRead))
                
                if file.endswith('.vtk'):
                    # Setting up reader able to read *.vtk files (change reader if necessary)
                    reader = vtk.vtkGenericDataObjectReader()
                elif file.endswith('.vtu'):
                    # Setting up reader able to read *.vtu files (change reader if necessary)
                    reader = vtk.vtkXMLUnstructuredGridReader()
                else:
                    print(f"ERROR IN {func_name}: Unsupported file extension")
                    sys.exit(1)

                # Data extraction
                reader.SetFileName(fileToRead)
                reader.Update()
                data = reader.GetOutput()
                try:
                    var = data.GetCellData().GetArray(fieldName)
                    var = vtk_np.vtk_to_numpy(var)
                except:
                    print(f"WARNING IN {func_name}: Array of this name does not exist in this file, skipping to next file")
                    continue
                
                if fieldType == 'vector':
                    if i == 0:        # Saving initial data and allocating output variable
                        vectorDim = 3
                        nRows = var.shape[0]
                        out = np.empty((vectorDim*nRows,1))   
                        # Reshaping data to desired output shape
                        varCol = np.reshape(var,vectorDim*nRows,order = order)      
                        out[:,0] = varCol
                    else:
                        varCol = np.reshape(var,(vectorDim*nRows,1),order = order)   # Reshaping data to desired output shape
                        out = np.append(out, varCol, axis=1)

                elif fieldType == 'scalar':
                    if i == 0:      # Saving initial data and allocating output variable
                        nRows = var.size
                        out = np.empty((nRows,1))
                        out[:,0] = var
                    else:
                        out = np.append(out, var.reshape(-1, 1), axis=1)

                else:
                    print(f"ERROR IN {func_name}: Unsupported field type")
                    sys.exit(1)

                i += 1
    
    # Saving data
    if saveMatrix:
        if verbose:
            print(" -> Saving data into " + str(savePath))
        if not os.path.exists(os.path.dirname(savePath)):
            os.makedirs(os.path.dirname(savePath))
        np.save(savePath, out)
        if verbose:
            print(" -> Done.")

    print("Done.")        
    return out




def readOneVtk(filePath,fieldName,fieldType='scalar',order='',saveMatrix=False,savePath='./oneVtk.npy',verbose=False):
    """
    Extracts data from specified vtk / vtu file and saves the data in matrix form.

    Parameters
    ----------
    filePath : str
        Path to the file to be extracted, absolute or relative.
    
    fieldName : str
        Name of the field to be extracted.

    fieldType : str, optional
        What type of field to expect. Can take on following values:
            'scalar' = scalar field (pressure, density, ...)
            'vector' = 3D vector field (velocity, vorticity, ...)
        Default 'scalar'.

    order : str, optional
        How to flatten vector field from its vtk representation.
        Can take on following values:
            'C': C-like ordering, last index changes fastest (flattened field looks like [u1_x, u1_y, u1_z, u2_x, ...])
            'F': Fortran-like ordering, first index changes fastest (flattened field looks like [u1_x, u2_x, ..., u1_y, u2_y, ...])
        Not required for scalar fields.

    saveMatrix : bool, optional
        Whether to save the matrix and return it (True) or only return the matrix (False).
        Default False.

    savePath : str, optional
        Where to save the data. Relevant only when saveMatrix == True. Default './oneVtk.npy'.
    
    verbose : bool, optional
        Provide aditional info about intermediate steps during execution. Default False.

    Returns
    -------
    out : 2D np.ndarray
        Output variable of size (nDim*nCells x 1). Contains all data from vtk in one column.
        This shape was chosen to make it consistent with 'readAll' function
    """

    print("Extracting %s field %s from %s ..."%(fieldType,fieldName,filePath))
    func_name = inspect.currentframe().f_code.co_name
    # type checks
    try:
        assert (isinstance(filePath,str) and os.path.exists(filePath)), "Variable 'filePath' is not string or is not a valid path"
        assert isinstance(fieldName,str), "Variable 'fieldName' must be a string"
        assert isinstance(order,str), "Variable 'order' must be a 1 character string"
        assert isinstance(saveMatrix,bool), "Variable 'saveMatrix' must be of type 'bool'"
        assert isinstance(savePath,str), "Variable 'savePath' must be a string"
        assert isinstance(verbose,bool), "Variable 'verbose' must be of type 'bool'"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        # Exit the program immediately
        sys.exit(1)


    if filePath.endswith('.vtk'):
        # Setting up reader able to read *.vtk files (change reader if necessary)
        reader = vtk.vtkGenericDataObjectReader()
    elif filePath.endswith('.vtu'):
        # Setting up reader able to read *.vtu files (change reader if necessary)
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        print(f"ERROR IN {func_name}: Unsupported file extension")
        sys.exit(1)

    # Data extraction
    reader.SetFileName(filePath)
    reader.Update()
    data = reader.GetOutput()
    try:
        var = data.GetCellData().GetArray(fieldName)
        var = vtk_np.vtk_to_numpy(var)
    except:
        print(f"ERROR IN {func_name}: Unsupported file extension")
        sys.exit(1)

    if fieldType == 'vector':
        vectorDim = 3
        nRows = var.shape[0]
        out = np.empty((vectorDim*nRows,1))   
        # Reshaping data to desired output shape
        varCol = np.reshape(var,vectorDim*nRows, order = order)      
        out[:,0] = varCol

    elif fieldType == 'scalar':
        nRows = var.size
        out = np.empty((nRows,1))
        out[:,0] = var

    else:
        print(f"ERROR IN {func_name}: Unsupported field type")
        sys.exit(1)
    
    # Save matrix
    if saveMatrix:
        if verbose:
            print(" -> Saving data into " + str(savePath))
        if not os.path.exists(os.path.dirname(savePath)):
            os.makedirs(os.path.dirname(savePath))
        np.save(savePath, out)
        if verbose:
            print(" -> Done.")

    print("Done.")        
    return out



def getCellVolumes(fileName,threshold=1e-24):
    """
    Extracts cell volumes from vtk, vtu and similar files.

    Parameters
    ----------
    fileName : str
        Path of the vtk or similar type file for which the volumes will be computed.

    threshold : float, optional
        Threshold value for finding zero-volume cells. If any cell has vol < threshold,
        the warning about zero volume cells in the mesh will be raised. Default 1e-24.
    
    Returns
    -------
    vol : 1D np.ndarray
        Output variable. Contains volumes of cells. 
        Volumes are listed in the same order as the cells are indexed in the vtk file.

    Notes
    -----
    USE WITH CAUTION AND DO NOT BLINDLY TRUST THE RESULTS
        Issues with the pyvista compute_cell_sizes() function have been reported in the past on some mesh types.
        For details see, e.g., https://github.com/pyvista/pyvista/issues/3708
        My implementation tries to bypass the negative volume issues, but extreme caution must be taken
        whenever negative volumes are reported.
    """

    print("Computing cell volumes in %s ..."%fileName)
    func_name = inspect.currentframe().f_code.co_name
    # type checks
    try:
        assert (isinstance(fileName,str) and os.path.exists(fileName)), "Variable 'fileName' is not string or is not a valid path"
    except AssertionError as e:
        # Print the error message
        print(f"ERROR IN {func_name}: {e}")
        # Exit the program immediately
        sys.exit(1)

    # Read the VTK file (supports .vtk, .vtu, .vtp, etc.)
    mesh = pv.read(fileName)

    if mesh.n_cells == 0:
        print(f"ERROR IN {func_name}: No cells found in the mesh")
        sys.exit(1)

    # Compute cell volumes
    vol = np.array(mesh.compute_cell_sizes(length=False, area=False, volume=True)["Volume"])
    # Check dimensions
    if vol.size != mesh.n_cells:
        print(f"ERROR IN {func_name}: No of cell volumes does not correspond to the no of cells")
        sys.exit(1)
    # Check for negative volumes
    if np.any(vol < 0):
        print(f"WARNING IN {func_name}: Negative volumes encountered, mesh may be invalid or have incorrect point ordering")
        print(" -> Trying to bypass the issue by returning absolute values of cell volumes")
        print(" -> User discretion necessary, use provided values with caution")
        vol = np.abs(vol)
    # Check for zero volumes
    if np.any(vol < threshold):
        print(f"WARNING IN {func_name}: One or more zero-volume cells in the mesh, mesh may be invalid")

    print("Done.")
    return vol


