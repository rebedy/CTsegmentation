# CTsegmentation

This repository contains the code for the segmentation in CT images. The code is implemented in Python and uses vtk and SimpleITK libraries. The code is tested on Ubuntu 18.04.

<br />

## Installation

To install the required libraries, run the following command:

```bash
conda env create -f environment.yml
conda activate ct_seg
```

<br />

## Usage

Check `args.py` for the arguments that can be passed to the code.

<br />

To convert the DICOM files to VTI format, run the following command:

```bash
python misc/dcm2vti.py
```

<br />

To train model to segment the CT images, run the following command:

```bash
python train.py
```
