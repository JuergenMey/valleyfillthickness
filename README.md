Estimate valley-fill thicknesses using an artificial neural network
approach. For details refer to:

Mey, J., D. Scherler, G. Zeilinger, and M. R. Strecker (2015), Estimating
the fill thickness and bedrock topography in intermontane valleys using
artificial neural networks, J. Geophys. Res.  Earth Surf., 120, 1ï¿½20,
doi:10.1002/2014JF003270.

This code uses MATLAB's Parallel Computing Toolbox and requires the 
following function libraries, which can be downloaded from the MATLAB file 
exchange:

     netlab - An open-source neural network toolbox.
TopoToolbox - A MATLAB program for the analysis of digital elevation
              models.

SYNTAX

[H,Z,E,STD] = vft(DEM,MASK,input,fraction,sectors,nodes,buffer,iterations,path)

DESCRIPTION

Computes a map of valley-fill thicknesses based on the geometric properties
of a digital elevation model (DEM) and a mask of the valley-fill. Output
is saved to 'CurrentFolder/out'(default).  
   

INPUT(required)

DEM     Digital elevation model (class: GRIDobj)
MASK    Mask of the valley fill containing only 0s and 1s with 1s indicating
        valley-fill cells (class: GRIDobj)
  
INPUT(optional)

input            'distance' --> use distances as network inputs (default)
                'elevation' --> use distances and elevations as inputs
              'coordinates' --> use distances and coordinates as inputs
    'elevation+coordinates' --> use distances, elevations and coordinates
                                as network inputs
fraction    Number of training cells in relation to number of fill cells,
            has a large impact on the computing time, (default: 0.1)
            
sectors     Range of directional sectors (default: [1 10])
nodes       Range of hidden nodes (default: [1 10])
buffer      Maximum distance between training cells and valley fill in
            meters (default: 2000)
iterations  Network learning cycles (default: 1000)
path        define output location  

OUTPUT

    H       Map of valley-fill thicknesses (class: GRIDobj)
    Z       Map of bedrock elevations (class: GRIDobj)
    E       Validation error as a function of the network configuration
    STD     Map of the standard deviation determined from all parallel runs
    NET     The trained network(s) that can be applied to other fillmasks 
            without retraining.           

USAGE

H = vft(DEM,MASK)
H = vft(DEM,MASK,'fraction',0.5)
[H,Z,E,STD] = vft(DEM,MASK,'input','coordinates','nodes',[1 20],'buffer',1000)

EXAMPLE 

DEM = GRIDobj('yosemite_valley.tif'); % ASTER GDEM
MASK = GRIDobj('fillmask.tif');  % from NPS GRI [2006]

[H,Z,E,STD] = vft(DEM,MASK,'path','fill','fraction',0.01,'sectors',[1 5],'nodes',[1 5],'buffer',1000);
imageschs(DEM);figure;imageschs(Z)
% compare with independent estimates from Gutenberg et al. [1956], Fig.10
D = GRIDobj('depthtobedrock.tif');
figure;imagesc(D-H)

References:
NPS Geologic Resources Inventory Program (2006). Glacial and Postglacial 
Deposits in Yosemite Valley, California (NPS, GRD, GRE, YOSE), Lakewood, CO.

Gutenberg, B., J. P. Buwalda, and R. P. Sharp (1956), Seismic explorations on 
the floor of Yosemite Valley, California, Bull. Geol. Soc. Am., 67, 1051ï¿½1078.
