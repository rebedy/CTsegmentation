import itk
import vtk
from vtk.util import numpy_support

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# -------------------------------------------------------       #

PixelType = itk.ctype("signed short")
# PixelType = itk.ctype("unsigned char")
# PixelType = itk.UCs
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]


def itkDicomReader_N_2Array(path):
    """
    To read dicom files using itk package and convert to array.
    :param path:  a directory with files of dicom series.
    """
    # NamesGenerator Takes too much time
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")  # (0008,0021)	DA	Series Date
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(path)
    seriesUID = namesGenerator.GetSeriesUIDs()
    main_uid, len_uid = 0, 0
    for i, uid in enumerate(seriesUID):
        namesGen = namesGenerator.GetFileNames(uid)
        # print('len(namesGen) : ', len(namesGen))
        if len(namesGen) >= len_uid:
            len_uid = len(namesGen)
            main_uid = uid
        else:
            pass
    fileNames = namesGenerator.GetFileNames(main_uid)  # seriesUID[0])

    # ### *DY| Read Dicom
    reader = itk.ImageSeriesReader[ImageType].New()  # This module allows more functions to be used.
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()
    itkImage = reader.GetOutput()

    array = itk.GetArrayFromImage(itkImage)

    return itkImage, array


def array2vtkImageData(itkImage, array):
    """
    Converting Nd-array to vtkImageData using vtk and vtk.numpy_support packages.
    :param itkImage:
    :param array:
    :param file_path:
    :param arr_type:
    """

    vtkImage = numpy_support.numpy_to_vtk(num_array=array.ravel(), deep=True, array_type=vtk.VTK_SHORT)

    vtkimageData = vtk.vtkImageData()
    vtkimageData.SetDimensions(array.shape[1], array.shape[2], array.shape[0])
    vtkimageData.GetPointData().SetScalars(vtkImage)
    vtkimageData.SetSpacing(list(itkImage.GetSpacing()))

    return vtkimageData


def vtkImageData2vtiWriter(vtkimageData, file_path):
    """
    Writing vtkImageData to .vti file format using vtk package.
    :param vtkimageData:
    :param file_path:
    """

    # ### * Writing .vti data
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(vtkimageData)
    writer.SetFileName(file_path)
    writer.Update()


def array2vtk_N_Write(array, file_path, spacing, arr_type):
    """
    Virtually a module of array2vtkImageData-vtkImageData2vtiWriter-combination.
    :param itkImage:
    :param array:
    :param file_path:
    :param arr_type:
    """

    vtkImage = numpy_support.numpy_to_vtk(num_array=array.ravel(), deep=True, array_type=arr_type)

    vtkimageData = vtk.vtkImageData()
    vtkimageData.SetDimensions(array.shape[1], array.shape[2], array.shape[0])
    vtkimageData.GetPointData().SetScalars(vtkImage)
    vtkimageData.SetSpacing(spacing)
    # print(vtkimageData.GetSpacing())
    # print(itkImage.GetLargestPossibleRegion().GetSize()) # itkSize3 ([512, 512, 187])

    # ### * Writing .vti datas
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(vtkimageData)
    writer.SetFileName(file_path)
    writer.Update()


def ChangePixelValueOfMask(arr_mask):
    '''
    Get first pixel value and change array values to:
        0 if ==first pixel value and
        1 if != first pixel value
    :param arr_mask:
    '''

    # ### *DY | To treat exported mask dataset from KHU
    for i in range(arr_mask.shape[0]):
        axial = arr_mask[i, :, :]
        val_first = axial[0, 0]
        axial = (axial != val_first).astype(int)  # (float)  ### if no astype, this is boolean.
        arr_mask[i, :, :] = axial

    return arr_mask


def vtkReader_N_2Array(vti_path):
    """
    Converting vtkImageXMLImageData to Nd-array using vtk and vtk.numpy_support packages.
    :param vti_path:
    """

    # ### Create the reader ###
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()

    # ### vtk2Numpy ###
    img = reader.GetOutput()
    dims = list(img.GetDimensions())  # (512, 512, 166~450)
    scalars = img.GetPointData().GetScalars()  # .GetRange() (-1024.0, 32767.0)
    np_data = numpy_support.vtk_to_numpy(scalars)
    np_data = np_data.reshape(dims[2], dims[0], dims[1])  # ndarray shape4img: H,W,C
    # print('np_data.shape',np_data.shape)

    return np_data


def vtkReader2Array_N_Spacing(vti_path):
    """
    Converting vtkImageXMLImageData to Nd-array using vtk and vtk.itk packages.
    :param vti_path:
    """

    # ### Create the reader ###
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()

    # ### vtk2Numpy ###
    img = reader.GetOutput()
    dims = list(img.GetDimensions())  # (512, 512, 166~450)
    scalars = img.GetPointData().GetScalars()  # .GetRange()) (0.0, 2.0)
    # print(img.GetPointData().GetScalars()
    np_data = numpy_support.vtk_to_numpy(scalars)
    np_data = np_data.reshape(-1, dims[0], dims[1])  # ndarray shape4img: H,W,C
    # print('np_data.shape',np_data.shape)
    spacing = img.GetSpacing()
    # print('spacing \n', spacing)

    return np_data, spacing
