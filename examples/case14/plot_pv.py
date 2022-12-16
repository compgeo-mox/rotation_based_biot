# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
import sys
from paraview.simple import *

def main(in_file_name, out_file_name):

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    sol_2_00000vtu = XMLUnstructuredGridReader(FileName=[in_file_name])
    sol_2_00000vtu.CellArrayStatus = ['p', 'p_ex', 'u', 'u_ex']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size

    # get layout
    layout1 = GetLayout()

    # show data in view
    sol_2_00000vtuDisplay = Show(sol_2_00000vtu, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    sol_2_00000vtuDisplay.Representation = 'Surface'
    sol_2_00000vtuDisplay.ColorArrayName = [None, '']
    sol_2_00000vtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    sol_2_00000vtuDisplay.SelectOrientationVectors = 'None'
    sol_2_00000vtuDisplay.ScaleFactor = 10.0
    sol_2_00000vtuDisplay.SelectScaleArray = 'None'
    sol_2_00000vtuDisplay.GlyphType = 'Arrow'
    sol_2_00000vtuDisplay.GlyphTableIndexArray = 'None'
    sol_2_00000vtuDisplay.GaussianRadius = 0.5
    sol_2_00000vtuDisplay.SetScaleArray = [None, '']
    sol_2_00000vtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    sol_2_00000vtuDisplay.OpacityArray = [None, '']
    sol_2_00000vtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    sol_2_00000vtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
    sol_2_00000vtuDisplay.PolarAxes = 'PolarAxesRepresentation'
    sol_2_00000vtuDisplay.ScalarOpacityUnitDistance = 18.589811721538794

    # reset view to fit data
    renderView1.ResetCamera()

    #changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [50.0, 5.0, 10000.0]
    renderView1.CameraFocalPoint = [50.0, 5.0, 0.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Plot Over Line'
    plotOverLine1 = PlotOverLine(Input=sol_2_00000vtu,
        Source='High Resolution Line Source')

    # init the 'High Resolution Line Source' selected for 'Source'
    plotOverLine1.Source.Point2 = [100.0, 10.0, 0.0]

    # show data in view
    plotOverLine1Display = Show(plotOverLine1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    plotOverLine1Display.Representation = 'Surface'
    plotOverLine1Display.ColorArrayName = [None, '']
    plotOverLine1Display.OSPRayScaleArray = 'arc_length'
    plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    plotOverLine1Display.SelectOrientationVectors = 'None'
    plotOverLine1Display.ScaleFactor = 10.0
    plotOverLine1Display.SelectScaleArray = 'None'
    plotOverLine1Display.GlyphType = 'Arrow'
    plotOverLine1Display.GlyphTableIndexArray = 'None'
    plotOverLine1Display.GaussianRadius = 0.5
    plotOverLine1Display.SetScaleArray = ['POINTS', 'arc_length']
    plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
    plotOverLine1Display.OpacityArray = ['POINTS', 'arc_length']
    plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
    plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
    plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    plotOverLine1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 100.4987564086914, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    plotOverLine1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 100.4987564086914, 1.0, 0.5, 0.0]

    # Create a new 'Line Chart View'
    lineChartView1 = CreateView('XYChartView')
    # uncomment following to set a specific view size
    # lineChartView1.ViewSize = [400, 400]

    # show data in view
    plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')

    # trace defaults for the display properties.
    plotOverLine1Display_1.CompositeDataSetIndex = [0]
    plotOverLine1Display_1.UseIndexForXAxis = 0
    plotOverLine1Display_1.XArrayName = 'arc_length'
    plotOverLine1Display_1.SeriesVisibility = ['p', 'p_ex', 'u_Magnitude', 'u_ex_Magnitude']
    plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', 'p', 'p', 'p_ex', 'p_ex', 'u_X', 'u_X', 'u_Y', 'u_Y', 'u_Z', 'u_Z', 'u_Magnitude', 'u_Magnitude', 'u_ex_X', 'u_ex_X', 'u_ex_Y', 'u_ex_Y', 'u_ex_Z', 'u_ex_Z', 'u_ex_Magnitude', 'u_ex_Magnitude', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
    plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'p', '0.89', '0.1', '0.11', 'p_ex', '0.22', '0.49', '0.72', 'u_X', '0.89', '0.1', '0.11', 'u_Y', '0.22', '0.49', '0.72', 'u_Z', '0.3', '0.69', '0.29', 'u_Magnitude', '0.6', '0.31', '0.64', 'u_ex_X', '1', '0.5', '0', 'u_ex_Y', '0.65', '0.34', '0.16', 'u_ex_Z', '0', '0', '0', 'u_ex_Magnitude', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
    plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', 'p', '0', 'p_ex', '0', 'u_X', '0', 'u_Y', '0', 'u_Z', '0', 'u_Magnitude', '0', 'u_ex_X', '0', 'u_ex_Y', '0', 'u_ex_Z', '0', 'u_ex_Magnitude', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
    plotOverLine1Display_1.SeriesLabelPrefix = ''
    plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', 'p', '1', 'p_ex', '1', 'u_X', '1', 'u_Y', '1', 'u_Z', '1', 'u_Magnitude', '1', 'u_ex_X', '1', 'u_ex_Y', '1', 'u_ex_Z', '1', 'u_ex_Magnitude', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
    plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', 'p', '2', 'p_ex', '2', 'u_X', '2', 'u_Y', '2', 'u_Z', '2', 'u_Magnitude', '2', 'u_ex_X', '2', 'u_ex_Y', '2', 'u_ex_Z', '2', 'u_ex_Magnitude', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
    plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', 'p', '0', 'p_ex', '0', 'u_X', '0', 'u_Y', '0', 'u_Z', '0', 'u_Magnitude', '0', 'u_ex_X', '0', 'u_ex_Y', '0', 'u_ex_Z', '0', 'u_ex_Magnitude', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
    plotOverLine1Display_1.SeriesMarkerSize = ['arc_length', '4', 'p', '4', 'p_ex', '4', 'u_X', '4', 'u_Y', '4', 'u_Z', '4', 'u_Magnitude', '4', 'u_ex_X', '4', 'u_ex_Y', '4', 'u_ex_Z', '4', 'u_ex_Magnitude', '4', 'vtkValidPointMask', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'Points_Magnitude', '4']

    # add view to a layout so it's visible in UI
    AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

    # update the view to ensure updated data information
    lineChartView1.Update()

    # update the view to ensure updated data information
    renderView1.Update()

    # update the view to ensure updated data information
    lineChartView1.Update()

    # Properties modified on plotOverLine1Display_1
    plotOverLine1Display_1.SeriesVisibility = ['p', 'p_ex', 'u_Magnitude', 'u_ex_Magnitude']
    plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'p', '0.889998', '0.100008', '0.110002', 'p_ex', '0.220005', '0.489998', '0.719997', 'u_X', '0.889998', '0.100008', '0.110002', 'u_Y', '0.220005', '0.489998', '0.719997', 'u_Z', '0.300008', '0.689998', '0.289998', 'u_Magnitude', '0.6', '0.310002', '0.639994', 'u_ex_X', '1', '0.500008', '0', 'u_ex_Y', '0.650004', '0.340002', '0.160006', 'u_ex_Z', '0', '0', '0', 'u_ex_Magnitude', '0.889998', '0.100008', '0.110002', 'vtkValidPointMask', '0.220005', '0.489998', '0.719997', 'Points_X', '0.300008', '0.689998', '0.289998', 'Points_Y', '0.6', '0.310002', '0.639994', 'Points_Z', '1', '0.500008', '0', 'Points_Magnitude', '0.650004', '0.340002', '0.160006']
    plotOverLine1Display_1.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'p', '0', 'p_ex', '0', 'u_Magnitude', '0', 'u_X', '0', 'u_Y', '0', 'u_Z', '0', 'u_ex_Magnitude', '0', 'u_ex_X', '0', 'u_ex_Y', '0', 'u_ex_Z', '0', 'vtkValidPointMask', '0']
    plotOverLine1Display_1.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'arc_length', '1', 'p', '1', 'p_ex', '1', 'u_Magnitude', '1', 'u_X', '1', 'u_Y', '1', 'u_Z', '1', 'u_ex_Magnitude', '1', 'u_ex_X', '1', 'u_ex_Y', '1', 'u_ex_Z', '1', 'vtkValidPointMask', '1']
    plotOverLine1Display_1.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'arc_length', '2', 'p', '2', 'p_ex', '2', 'u_Magnitude', '2', 'u_X', '2', 'u_Y', '2', 'u_Z', '2', 'u_ex_Magnitude', '2', 'u_ex_X', '2', 'u_ex_Y', '2', 'u_ex_Z', '2', 'vtkValidPointMask', '2']
    plotOverLine1Display_1.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'p', '0', 'p_ex', '0', 'u_Magnitude', '0', 'u_X', '0', 'u_Y', '0', 'u_Z', '0', 'u_ex_Magnitude', '0', 'u_ex_X', '0', 'u_ex_Y', '0', 'u_ex_Z', '0', 'vtkValidPointMask', '0']
    plotOverLine1Display_1.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'arc_length', '4', 'p', '4', 'p_ex', '4', 'u_Magnitude', '4', 'u_X', '4', 'u_Y', '4', 'u_Z', '4', 'u_ex_Magnitude', '4', 'u_ex_X', '4', 'u_ex_Y', '4', 'u_ex_Z', '4', 'vtkValidPointMask', '4']

    # Properties modified on plotOverLine1Display_1
    plotOverLine1Display_1.SeriesVisibility = ['p', 'p_ex']

    # save data
    SaveData(out_file_name, proxy=plotOverLine1, PointDataArrays=['arc_length', 'p', 'p_ex', 'u', 'u_ex', 'vtkValidPointMask'], Precision=15)

    #### saving camera placements for all active views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [50.0, 5.0, 10000.0]
    renderView1.CameraFocalPoint = [50.0, 5.0, 0.0]
    renderView1.CameraParallelScale = 50.24937810560445

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).

if __name__ == "__main__":

    main_folder = "/home/elle/Dropbox/Work/PresentazioniArticoli/2022/projects/rotation_based_biot/rotation_based_biot/examples/case14/"

    in_folder = main_folder + "sol/"

    in_file_name = sys.argv[1] #"sol_2_0000.0.vtu"
    out_file_name = sys.argv[2] #"data.csv"

    main(in_folder + in_file_name, out_file_name)
