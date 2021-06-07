# Vulkan-DICOM-viewer

The Vulkan DICOM viewer is a program that exports 3D models in stl, gltf and vtk formats when given a directory with a set of DICOM images. It also has an inbuilt rendering engine capable of generating a 3d model. The model data can be segmented by specifying a threshold parameter where generating the skeletal system from CT DICOM data requires a threshold parameter of 500. A demonstration of the program is shown in the youtube link below.

To install this program, you will need to know how to use CMake. You will also need to separately install the glm library and paste it in the external directory before configuring the application. ITK and VTK ( version 8.7 or higher) libaries need to be previously installed as well and the bin directories need to be pasted in the CMake GUI when the application is being configured. Ensure that ITKVTKGlue is configured when installing the ITK libraries otherwise the application will not work. This application is meant to run only on Windows operating systems and is not supported on any other OS. Use Microsoft Visual Studio to build the application. 

https://www.youtube.com/watch?v=AASvr6nWXcQ&ab_channel=ZachStarZachStarVerified
