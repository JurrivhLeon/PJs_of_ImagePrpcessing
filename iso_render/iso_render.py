import vtk
import nibabel as nib


# Read in the data.
def read_nib(path='./data/image_lr.nii.gz'):
    # Read in the scalar field data.
    heart_nib = nib.load(path)
    heart_voxel = heart_nib.get_fdata()
    print(f'Shape: {heart_voxel.shape}')

    # Construct the vtk image data.
    heart = vtk.vtkImageData()
    heart.SetDimensions(heart_voxel.shape)
    heart.SetSpacing(heart_nib.header['pixdim'][1:4])
    heart.SetOrigin(0, 0, 0)
    heart.AllocateScalars(vtk.VTK_DOUBLE, 1)
    for z in range(heart_voxel.shape[2]):
        for y in range(heart_voxel.shape[1]):
            for x in range(heart_voxel.shape[0]):
                heart.SetScalarComponentFromDouble(x, y, z, 0, heart_voxel[x, y, z])
    return heart


# Compute the isosurface.
def isosurface_compute(data, iso=150):
    # Use marching cube to compute the isosurface.
    extractor = vtk.vtkMarchingCubes()
    extractor.SetInputData(data)
    extractor.SetValue(0, iso)

    # Apply Laplacian smoothing.
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(extractor.GetOutputPort())
    smoother.SetRelaxationFactor(0.01)
    smoother.SetNumberOfIterations(300)

    # Generate triangle strips.
    # Not smoothed.
    stripper1 = vtk.vtkStripper()
    stripper1.SetInputConnection(extractor.GetOutputPort())
    # Smoothed.
    stripper2 = vtk.vtkStripper()
    stripper2.SetInputConnection(smoother.GetOutputPort())
    return stripper1, stripper2


# Render the isosurface.
def render(stripper):
    # Render the isosurface.
    # Maps data to graphic primitives.
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())

    # Set the actor.
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 1, 0)
    actor.GetProperty().SetOpacity(0.95)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.6)
    actor.GetProperty().SetSpecular(1.0)

    # Set the renderer, window, and interactor.
    renderer = vtk.vtkRenderer()
    renderer.SetBackground((1, 1, 1))
    renderer.AddActor(actor)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    # Render.
    interactor.Initialize()
    window.Render()
    interactor.Start()


if __name__ == '__main__':
    heart = read_nib(path='./data/image_lr.nii.gz')
    isosurface, isosurface_smoothed = isosurface_compute(heart, iso=150)
    render(isosurface)
    render(isosurface_smoothed)
