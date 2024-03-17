import  itk

# ------------------------------------------------------- #


ImageType_SS = itk.Image[itk.ctype('signed short'), 3]  # -32768~32767
ImageType_F = itk.Image[itk.ctype('float'), 3]  # 3.4E-38(-3.4*10^38) ~ 3.4E+38(3.4*10^38) (7digits)
ImageType_SS_2D = itk.Image[itk.ctype('signed short'), 2]


def ThresholdItkImage(itkImage, below, upper):
    '''
    itkImage type: signed short
    '''

    thresholder = itk.ThresholdImageFilter[ImageType_SS].New()
    thresholder.SetInput(itkImage)
    thresholder.ThresholdBelow(below)
    thresholder.SetOutsideValue(below)
    thresholder.Update()

    upperThresholder = itk.ThresholdImageFilter[ImageType_SS].New()
    upperThresholder.SetInput(thresholder.GetOutput())
    upperThresholder.ThresholdAbove(upper)
    upperThresholder.SetOutsideValue(0)
    upperThresholder.Update()
    itkImage_threshold = upperThresholder.GetOutput()

    return itkImage_threshold


def NormalizeItkImage(itkImage, normalization):
    '''
    itkImage type: input-type of input, output-float
    '''

    rescaler = itk.RescaleIntensityImageFilter[type(itkImage), ImageType_F].New()
    rescaler.SetOutputMinimum(0.0)
    rescaler.SetOutputMaximum(float(normalization))
    rescaler.SetInput(itkImage)
    rescaler.Update()
    itkImage_normalized = rescaler.GetOutput()

    return itkImage_normalized
