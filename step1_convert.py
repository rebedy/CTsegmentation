import os
from os.path import join
import datetime
from glob import glob
import numpy as np

import itk
import vtk

from step0_args import *
# from step4_vti_rendering import *
# from data_io.itk_dicomIO import itkDicomIO, itkDicomIOMask, itkDicomIOMaskLoop
from data_io.vti_processing import itkDicomReader_N_2Array, array2vtkImageData, array2vtk_N_Write, ChangePixelValueOfMask
from data_io.dcm_preprocessing import ThresholdItkImage, NormalizeItkImage
from misc.utils import *
# ------------------------------------------------------- #
np.set_printoptions(threshold=1000)  # sys.maxsize)

### ! Args Setting ! ###
TODAY = str(datetime.date.today().strftime('%y%m%d'))
NOW = str(datetime.datetime.now().strftime('_%Hh%Mm'))

SOURCE = SOURCE + DTYPE + "/" + SOURCE_COMPANY + "/" + SOURCE_COMPANY + "_exported/"

time_start = datetime.datetime.now()
print('\n')
log_text_name = join(SOURCE.split('/')[0], SOURCE_COMPANY + "_vti_converting_" + TODAY + NOW + ".txt")
log_text = open(log_text_name, 'a')
print('>>> log_text_name : ', log_text_name)
print('\n\r', file=log_text)

dir_list = glob(SOURCE + "/*/")
print(dir_list[:3])

for itr, pati_dir in enumerate(dir_list):
    keywords = os.listdir(pati_dir)
    dcm_dir = join(SOURCE, pati_dir.split("\\")[-2])
    target_dir = makedirs_ifnot(join(CONVERTED_DIR, pati_dir.split("\\")[-2]))

    # ### * 1. CT image Converting
    image_itkImage, image_arr = itkDicomReader_N_2Array(join(dcm_dir, keywords[0]))  # *** image_path
    itk_spacing = list(image_itkImage.GetSpacing())
    image_vtiName = join(target_dir, keywords[0] + ".vti")
    array2vtk_N_Write(image_arr, image_vtiName, itk_spacing, vtk.VTK_SHORT)

    # ### * 2. Normalized CT image Converting
    image_itkImage_threshold = ThresholdItkImage(image_itkImage, below=-1000, upper=2500)
    image_itkImage_normalized = NormalizeItkImage(image_itkImage_threshold, 1)
    image_arr_norm = itk.GetArrayFromImage(image_itkImage_normalized)
    image_vtiName_norm = join(target_dir, keywords[0] + "_norm.vti")
    array2vtk_N_Write(image_arr_norm, image_vtiName_norm, itk_spacing, vtk.VTK_FLOAT)

    # ### * <Mask Image Combining and Converting>
    # maskData = itkDicomIOMask(key_dir) ### * No Need
    # maskData = itkDicomIOMaskLoop(mask_path_list)

    # ### * 3. Mask of Maxilla
    maxilla_itkImage, maxilla_arr = itkDicomReader_N_2Array(join(dcm_dir, keywords[1]))
    ''' #To Check arr
        # print(maxilla_arr.shape) # (187, 512, 512)
        ### *DY | To process exported mask dataset from YYY
        first = []
        for i in range(2):#maxilla_arr.shape[0]-1,maxilla_arr.shape[0]):
            axial = maxilla_arr[i, :, :]
            print(sum(sum(axial)))
            print(np.unique(axial))
            print('\n')
            val_first = axial[0,0]
            first.append(val_first)
            axial = (axial!=val_first).astype(float) ### if no astype to float, this is boolean.
            print(sum(sum(axial)))
            print(np.unique(axial))
            print('\n')
            maxilla_arr[i,:,:] = axial
        print(np.unique(first)) # [25079]
    '''
    maxilla_arr = ChangePixelValueOfMask(maxilla_arr)
    maxilla_vtkimageData = array2vtkImageData(image_itkImage, maxilla_arr)

    # ### * 4. Mask of Mandible
    mandible_itkImage, mandible_arr = itkDicomReader_N_2Array(join(dcm_dir, keywords[2]))
    mandible_arr = ChangePixelValueOfMask(mandible_arr)
    # print(sum(sum(sum(maxilla_arr))))   # 1,113,537
    # print(sum(sum(sum(mandible_arr))))  # 281,227   # 281,227+1,113,537 = 1,394,764

    # ### Write Each Mask
    maxilla_vtiName = join(target_dir,"maxilla.vti")
    array2vtk_N_Write(maxilla_arr, maxilla_vtiName, itk_spacing, vtk.VTK_UNSIGNED_CHAR)
    mandible_vtiName = join(target_dir,"mandible.vti")
    array2vtk_N_Write(mandible_arr, mandible_vtiName, itk_spacing, vtk.VTK_UNSIGNED_CHAR)
    # vtk.VTK_UNSIGNED_CHAR)
    # vtk.VTK_SHORT)

    # ### * 5. Merge the masks and Converting
    mandible_arr[mandible_arr == 1] = 2   # 2 for mandible mask value
    mask_arr = maxilla_arr + mandible_arr
    mask_arr[mask_arr == 3] = 0   # overlay pixels to maxilla(value 1)

    print('shape: ', mask_arr.shape, file=log_text)
    print('unique: ', np.unique(mask_arr), file=log_text)
    if all(np.unique(mask_arr) != [0, 1, 2]):
        print('Mask wrong!  ', pati_dir)
        print('unique: ', np.unique(mask_arr))
    else:
        pass
    mask_vtiName = join(target_dir, "mask.vti")
    array2vtk_N_Write(mask_arr, mask_vtiName, itk_spacing, vtk.VTK_SHORT)

    """
    if PREPROCESSED:
        ### Get spacing and thickness from itkImage spacing([spacing, spacing, thickness])
        spacing_list_vector = image_itkImage.GetSpacing().GetVnlVector()
        spacing_list = itk.GetArrayFromVnlVector(spacing_list_vector)
        spacing_original = spacing_list[0]
        thickness_original = spacing_list[2]
        # print('spacing_original: ', spacing_original)
        # print('thickness_original: ', thickness_original)

        ### Get size
        arr_ct_original = itk.GetArrayFromImage(itkImage_ct)
        size_original = np.copy(arr_ct_original.shape)
        # print('arr_ct_original: ', arr_ct_original.shape, np.min(arr_ct_original), np.max(arr_ct_original))
    """

    print(str(itr) + '   ' + target_dir)
    print(str(itr) + '   ' + target_dir + '\n\r', file=log_text)
