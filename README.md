This matlab code uses a 2-layer feed forward neural network to estimate valley fill thicknesses.

Setup:
1. Download/clone the content of the repository
2. Add netlab to your Matlab path.
3. Run the example.
     DEM = GRIDobj('yosemite_valley.tif'); % ASTER GDEM
     MASK = GRIDobj('fillmask.tif');  % from NPS GRI [2006]

     [H,Z,E,STD] = vft(DEM,MASK,'path','fill','fraction',0.01,'sectors',[1 5],'nodes',[1 5],'buffer',1000);
     imageschs(DEM);figure;imageschs(Z)
     % compare with independent estimates from Gutenberg et al. [1956], Fig.10
     D = GRIDobj('depthtobedrock.tif');
     figure;imagesc(D-H)
