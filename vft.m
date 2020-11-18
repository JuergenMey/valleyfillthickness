function [H,varargout] = vft(DEM,MASK,varargin)
% Estimate valley-fill thicknesses using an artificial neural network
% approach. For details refer to:
%
% Mey, J., D. Scherler, G. Zeilinger, and M. R. Strecker (2015), Estimating
% the fill thickness and bedrock topography in intermontane valleys using
% artificial neural networks, J. Geophys. Res.  Earth Surf., 120, 1�20,
% doi:10.1002/2014JF003270.
%
% This code uses MATLAB's Parallel Computing Toolbox and requires the 
% following function libraries, which can be downloaded from the MATLAB file 
% exchange:
%
%      netlab - An open-source neural network toolbox.
% TopoToolbox - A MATLAB program for the analysis of digital elevation
%               models.
%
% SYNTAX
%
% [H,Z,E,STD] = vft(DEM,MASK,input,fraction,sectors,nodes,buffer,iterations)
%
% DESCRIPTION
%
% Computes a map of valley-fill thicknesses based on the geometric properties
% of a digital elevation model (DEM) and a mask of the valley-fill. Output
% is saved to 'CurrentFolder/out'(default).  
%    
%
% INPUT(required)
%
% DEM     Digital elevation model (class: GRIDobj)
% MASK    Mask of the valley fill containing only 0s and 1s with 1s indicating
%         valley-fill cells (class: GRIDobj)
%   
% INPUT(optional)
%
% input            'distance' --> use distances as network inputs (default)
%                 'elevation' --> use distances and elevations as inputs
%               'coordinates' --> use distances and coordinates as inputs
%     'elevation+coordinates' --> use distances, elevations and coordinates
%                                 as network inputs
% fraction    Number of training cells in relation to number of fill cells,
%             has a large impact on the computing time, (default: 0.1)
%             
% sectors     Maximum number of directional sectors (default: 10)
% nodes       Maximum number of hidden nodes (default: 10)
% buffer      Maximum distance between training cells and valley fill in
%             meters (default: 2000)
% iterations  Network learning cycles (default: 1000)
% path        define output location  
%
% OUTPUT
%
%     H       Map of valley-fill thicknesses (class: GRIDobj)
%     Z       Map of bedrock elevations (class: GRIDobj)
%     E       Validation error as a function of the network configuration
%     STD     Map of the standard deviation determined from all parallel runs
%               
%
% USAGE
%
% H = vft(DEM,MASK)
% H = vft(DEM,MASK,'fraction',0.5)
% [H,Z,E,STD] = vft(DEM,MASK,'input','coordinates','nodes',20,'buffer',1000)
%
% EXAMPLE 
%
% DEM = GRIDobj('yosemite_valley.tif'); % ASTER GDEM
% MASK = GRIDobj('fillmask.tif');  % from NPS GRI [2006]
% 
% [H,Z,E,STD] = vft(DEM,MASK,'path','fill','fraction',0.01,'sectors',5,'nodes',5,'buffer',1000);
% imageschs(DEM);figure;imageschs(Z)
% % compare with independent estimates from Gutenberg et al. [1956], Fig.10
% D = GRIDobj('depthtobedrock.tif');
% figure;imagesc(D-H)
%
% References:
% NPS Geologic Resources Inventory Program (2006). Glacial and Postglacial 
% Deposits in Yosemite Valley, California (NPS, GRD, GRE, YOSE), Lakewood, CO.
% 
% Gutenberg, B., J. P. Buwalda, and R. P. Sharp (1956), Seismic explorations on 
% the floor of Yosemite Valley, California, Bull. Geol. Soc. Am., 67, 1051�1078.
%
%
% Tested with MATLAB 2012b
% Author: J�rgen Mey (mey[at]geo.uni-potsdam.de)
% Date: 03. January, 2016

% close any existing parallel sessions
delete(gcp('nocreate'))
 
% open matlabpool for parallel computing
try
    parpool('local',feature('numCores'));
catch 
    disp('Parallel Computing Toolbox is not installed. Running code in serial mode...')
end

nargoutchk(1,4);
narginchk(2,16)

% only want 12 optional inputs at most
numvarargs = length(varargin);
if numvarargs > 14
    error('vft:TooManyInputs', ...
        'requires at most 7 optional inputs');
end

p = inputParser;
defaultInput = 'distance';
expectedInput = {'distance','elevation','coordinates','elevation+coordinates','path'};
defaultFraction = 0.1;
defaultSectors = 10;
defaultNodes = 10;
defaultBuffer = 2000;
defaultIterations = 1000;
defaultpath = 'out';

addRequired(p,'DEM',@(x)isa(x,'GRIDobj'));
addRequired(p,'MASK',@(x)isa(x,'GRIDobj'));
addParameter(p,'input',defaultInput,...
    @(x) any(validatestring(x,expectedInput)));
addParameter(p,'fraction',defaultFraction,@isnumeric);
addParameter(p,'sectors',defaultSectors,@isnumeric);
addParameter(p,'nodes',defaultNodes,@isnumeric);
addParameter(p,'buffer',defaultBuffer,@isnumeric);
addParameter(p,'iterations',defaultIterations,@isnumeric);
addParameter(p,'path',defaultpath);
                 
parse(p,DEM,MASK,varargin{:});

switch p.Results.input
    case 'distance'
        addinput = 0;
    case 'elevation'
        addinput = 1;
    case 'coordinates'
        addinput = 2;
    case 'elevation+coordinates'
        addinput = 3;
end

fexamples = p.Results.fraction;
train_buffer = p.Results.buffer;
maxsector = p.Results.sectors;
maxnodes = p.Results.nodes;
niterations = p.Results.iterations;
outdirection = p.Results.path;

%% Input definition
tic;   

 

% Configuration
nrun                = 1 : 4;    % interval of test runs (1 : #CPU cores)
ridge_buffer        = 1;        % minimum Euclidean distance between training cells and ridges
validate            = 1;        % perform validation
threshold           = 1;        % to adjust sampling of training thicknesses


% Network parameters
valexamples         = fexamples;    % fraction of potential training cells used for validation, (valexamples + maxexamples <= 1)
numnet              = 3;            % number of networks to train, network with lowest training error is selected for prediction
minsector           = 1;            % minimum number of sectors
minnodes            = 1;            % minimum number of hidden nodes
%% PREPROCESSING
MASK.Z(isnan(MASK.Z)) = 0;
DEMc = DEM;MASKc=MASK;


%slope mask
slope = gradient8(DEMc,'deg');

%hydrology
DEMc_fill = fillsinks(DEMc);
FlowDir = FLOWobj(DEMc_fill);
FlowAcc = flowacc(FlowDir);

%ridge mask
Ridges = zeros(size(FlowAcc.Z));
Ridges(FlowAcc.Z == 1) = 1;
Ridges = medfilt2(Ridges);
Ridges = medfilt2(Ridges);
Ridges = medfilt2(Ridges);
Ridges = medfilt2(Ridges);
Ridges = bwareaopen(Ridges,15);

%prepare Train_data and mask_test
Train_data = DEMc.Z;
ica = find(MASKc.Z == 1);       % find cells belonging to the valley-fill
Train_catch = Train_data;
Train_data(slope.Z < 10) = 0;   % exclude flat areas from training data
Train_data(MASKc.Z == 1) = 0;

% define maximum distance from a training cell to the fill
[V,~] = bwdist(MASKc.Z);
V = V.*DEMc.cellsize;
Train_data(V>train_buffer) = 0;
Train_catch(V>train_buffer) = 0;

%slicing variables for parallel computing
DEMcZ = DEMc.Z;
MASKcZ = MASKc.Z;
Ridges(Train_data == 0) = 0;                  % set Ridges inside Train_data to 0

[D,~] = bwdist(Ridges);                       % Distance to (D)/ position of (IDX) nearest ridge
mkdir(['.\',num2str(outdirection)])

%% MAIN
% initialize storages
RESULTSv =zeros(maxnodes,maxsector,8,nrun(end));
NET = cell(maxnodes,maxsector,4);
Distance_train = cell(maxsector,4);
Target_train = cell(maxsector,4);
Distance_val = cell(maxsector,4);
Target_val = cell(maxsector,4);
texamples = floor(fexamples*size(ica,1));
vexamples = floor(valexamples*(size(ica,1)));
ic = find(Train_data ~= 0  & D >= ridge_buffer);                        % find cells of Train_data that fall outside ridge_buffer distance

parfor testnum = nrun
    mask = zeros(size(Train_data));
    y = randsample(ic,texamples);                                       % randomly sample texamples cells from the pool of potential training cells
    mask(y) = Train_data(y);                                            % mask of training cells
    nrange_crate = zeros(texamples,1);
    % calculate the maximum possible training thickness for each element of y
    disp('compute range of possible training thicknesses')
    for n = 1 : texamples
        nridge = Ridges;
        nridge(DEMcZ <= Train_data(y(n))) = 0;
        [~,IDX]=bwdist(nridge);
        nrange =  DEMcZ(IDX(y(n))) - Train_data(y(n));                  % elevation range
        nrange_crate(n,1) = nrange;
    end
    id = find(nrange_crate >= threshold);                               % apply threshold
    if size(id,1)<texamples
        id = randsample(id,texamples,true);
    end
    if validate == 1
        % set up the validation set
        Train_datav = Train_data;
        Train_datav(y) = 0;
        ie = find(Train_datav ~= 0 &  D >= ridge_buffer);
        if size(ie,1) < vexamples
            z = randsample(ie,vexamples,true);
        else
            z = randsample(ie,vexamples);
        end
        
        nrange_cratev = zeros(vexamples,1);
        % calculate the maximum possible validation thickness for each element of z
        disp('compute range of possible validation thicknesses')
        for n = 1 : vexamples
            nridge = Ridges;
            nridge(DEMcZ <= Train_data(z(n))) = 0;
            [~,IDX]=bwdist(nridge);
            nrange =  DEMcZ(IDX(z(n))) - Train_data(z(n));              % elevation range
            nrange_cratev(n,1) = nrange;
        end
        
        ik = find(nrange_cratev >= threshold);
        if size(ik,1)<vexamples
            ik = randsample(ik,vexamples,true);
        end
    end
    disp('compute distances for training/validation')
    for nsector = minsector : maxsector
        Storage_Train = zeros(texamples,nsector+addinput);              % Distances will be stored here and
        Target_Train = zeros(texamples,1);                              % corresponding thicknesses here
        for n = 1 : texamples
            ntarget = randsample(1:nrange_crate(id(n)),1);              % randomly sample 1 out of elevation range and
            train_elevation = Train_data(y(id(n))) + ntarget;           % add it to the training cell elevation to construct a training fill(=target)
            mask_train = zeros(size(Train_data));
            mask_train(Train_catch <= train_elevation & Train_catch ~= 0) = 1; % find cells in catchment < train elevation                                            % create 0,1-mask
            [i,k] = ind2sub(size(mask_train),y(id(n)));
            [nrows,ncols] = size(mask_train);
            [IX,IY] = meshgrid(1:ncols,1:nrows);
            J = -pi:pi/(nsector/2):pi;                                  % sector intervals from -pi to pi
            iX = IX - k;
            iY = IY - i;
            [THETA,RHO] = cart2pol(iX,iY);
            
            %compute distances for each sector
            try
                for m = 1 : nsector
                    Storage_Train(n,m) = min(RHO(THETA>=J(m) & THETA<J(m+1) & mask_train==0 ));
                    if addinput == 1 || addinput == 3
                        Storage_Train(n,end-addinput+1)= train_elevation;
                    end
                end
                Target_Train(n,1) = ntarget;
            catch                                                           % a cell that does not "see" a side wall in any direction
                Storage_Train(n,:) = 0;                                     % will cause an error and gets excluded from the training data set
                Target_Train(n,1) = 0;
            end
        end
        
        nn = find(Storage_Train(:,1) == 0);
        Target_Train = Target_Train(any(Storage_Train,2),:);
        Storage_Train = Storage_Train(any(Storage_Train,2),:);              % delete zero-rows
        
        % Include x-y-coordinates into training input
        if addinput == 2 || addinput == 3
            [ycord,xcord] = ind2sub(size(mask),y(id));
            ycord(nn) = 0; xcord(nn) = 0;
            Storage_Train(:,end-1) = ycord(ycord ~= 0);
            Storage_Train(:,end) = xcord(xcord ~= 0);
        end
        
        Distance_train{nsector,testnum} = Storage_Train;
        Target_train{nsector,testnum} = Target_Train;
        
        % normalisation
        Distance_train_norm = [];
        for col = 1 : size(Distance_train{nsector,testnum},2)
            mean_Distance_train = mean(Distance_train{nsector,testnum}(:,col));
            std_Distance_train = std(Distance_train{nsector,testnum}(:,col));
            Distance_train_norm(:,col) = (Distance_train{nsector,testnum}(:,col) - mean_Distance_train) / std_Distance_train;
        end
        mean_Thickness_train = mean(Target_train{nsector,testnum});
        std_Thickness_train = std(Target_train{nsector,testnum});
        Thickness_train_norm = (Target_train{nsector,testnum} - mean_Thickness_train) / std_Thickness_train;
        
        % VALIDATION
        if validate == 1
            Storage_Val = zeros(vexamples,nsector+addinput);            % Distances are stored here and
            Target_Val = zeros(vexamples,1);                            % corresponding target thicknesses here
            for n = 1:vexamples
                ntarget = randsample(1:nrange_cratev(ik(n)),1);         % randomly sample 1 out of elevation range and
                train_elevation = Train_datav(z(ik(n))) + ntarget;      % add it to the train data elevation to construct a training fill(=target)
                mask_vald = zeros(size(Train_datav));
                mask_vald(Train_catch <= train_elevation & Train_catch ~= 0) = 1;  % find cells in catchment < train elevation                                           % create 0,1-mask
                [i,k] = ind2sub(size(mask_vald),z(ik(n)));
                [nrows,ncols] = size(mask_vald);
                [IX,IY] = meshgrid(1:ncols,1:nrows);
                J = -pi:pi/(nsector/2):pi;                              % sector intervals from -pi to pi
                iX = IX - k;
                iY = IY - i;
                [THETA,RHO] = cart2pol(iX,iY);
                
                %compute distances for each sector
                try
                    for m = 1 : nsector
                        Storage_Val(n,m) = min(RHO(THETA>=J(m) & THETA<J(m+1) & mask_vald==0 ));
                        if addinput == 1 || addinput == 3
                            Storage_Val(n,end-addinput+1) = train_elevation;
                        end
                    end
                    Target_Val(n,1) = ntarget;
                catch                                                       % a cell that does not "see" a side wall in any direction
                    Storage_Val(n,:) = 0;                                   % will cause an error and gets excluded from the training data set
                    Target_Val(n,1) = 0;
                end
            end
            nn = find(Target_Val==0);
            Target_Val = Target_Val(any(Storage_Val,2),:);
            Storage_Val = Storage_Val(any(Storage_Val,2),:);                % delete zero-rows
            
            % Include x-y-coordinates as training inputs
            if addinput == 2 || addinput == 3
                [ycord,xcord] = ind2sub(size(mask_vald),z(ik));
                ycord(nn) = 0;
                xcord(nn) = 0;
                ycord = ycord(any(ycord,2));
                xcord = xcord(any(xcord,2));
                Storage_Val(:,end-1) = ycord;
                Storage_Val(:,end) = xcord;
            end
            
            Distance_val{nsector,testnum} = Storage_Val;                % store for later usage
            Target_val{nsector,testnum} = Target_Val;
            
            % normalisation
            Distance_Val_norm = zeros(size(Storage_Val));
            for col = 1 : size(Distance_val{nsector,testnum},2)
                mean_Distance_train = mean(Distance_train{nsector,testnum}(:,col));
                std_Distance_train = std(Distance_train{nsector,testnum}(:,col));
                Distance_Val_norm(:,col) = (Distance_val{nsector,testnum}(:,col) - mean_Distance_train) / std_Distance_train;
            end
        end
        
        
        for nnodes = minnodes:maxnodes
            disp(['optimize configuration ' num2str(nsector) ' ' num2str(nnodes)])
            optdat1 = zeros(numnet,1);                                  % initialize optimization error storage
            optdat2 = cell(numnet,1);                                   % initialize network storage
            
            % TRAINING
            for it = 1:numnet                                           % loop over number of nets to train with this combination
                options = [0 10^-12 10^-12 0.0000001 0 0 0 0 0 0 0 0 0 niterations 0 0.0000001 0.1 1];
                net = mlp(nsector+addinput,nnodes,1,'linear');          % construct MLP
                [net, ~, varout] = netopt(net,options,Distance_train_norm,Thickness_train_norm,'scg');   % optimization procedure
                optdat1(it,1) = varout(end);
                optdat2{it,1} = net;
            end
            [o,~] = find(optdat1==min(optdat1));                        % find minimum optimization error and
            net = optdat2{o};                                           % select the corresponding network
            NET{nnodes,nsector,testnum} = net;                          % save the trained network
            minvarargout = min(optdat1);
            
            % VALIDATION
            if validate == 1
                predV = mlpfwd(net,Distance_Val_norm);                      % prediction on validation examples
                predTv = predV*std_Thickness_train + mean_Thickness_train;  % retransform values
                CorrCoeffv = corrcoef(predTv,Target_Val);                   % correlation coefficient of prediction and targets
                CorrCoeffv = CorrCoeffv(1,2);
                RMSEv = sqrt(mean((predTv-Target_Val).^2));                 % root mean squared error
                NRMSEv = RMSEv/(max(Target_Val)-min(Target_Val));           % root mean squared error normalized by the range of validation targets
                MDevv = mean(Target_Val-predTv);                            % mean deviation
                MedDevv = median(Target_Val-predTv);                        % median deviation
                dVv = (sum(Target_Val)-sum(predTv))/(sum(Target_Val)/100);  % delta volume
                RESULTSv(nnodes,nsector,:,testnum) = [nsector,nnodes,NRMSEv,MedDevv,RMSEv,CorrCoeffv,MDevv,dVv]; % validation results
            end
        end
    end
end

%%  PREDICTION

% find network configuration that performed best on the validation data set
mRESULTSv = mean(RESULTSv(:,:,3,:),4);
[nnodes,nsector] = find(mRESULTSv == min(mRESULTSv(:)));

% input generation
[nrows,ncols] = size(MASKc.Z);
[nrow,ncol] = find(MASKc.Z);
[IX,IY] = meshgrid(1:ncols,1:nrows);
J = -pi:pi/(nsector/2):pi;
StorageFill = zeros(size(nrow,1),nsector + addinput);                       % create result matrices for sectors
%compute distances
parfor i = 1 : size(nrow,1)
    disp(['compute distances for fill cells ' num2str(i/size(nrow,1)) '/1'])
    r = nrow(i);
    k = ncol(i);
    iX = IX - k;
    iY = IY - r;
    [THETA,RHO] = cart2pol(iX,iY);
    try
        for m = 1 : nsector
            StorageFill(i,m) = min(RHO(THETA>=J(m) & THETA<J(m+1) & MASKcZ==0 ));
        end
    catch error1                                            % ignore cells that cause an error
        %                 for m = 1 : nsector
        %                     StorageFill(i,m) = 0;
        %                 end
    end
end


dt = find(StorageFill(:,1)~=0);
Distance_test = StorageFill(any(StorageFill,2),:);              % delete zero-rows

% include elevation as input
if addinput == 1 || addinput == 3
    Distance_test(:,end-addinput+1) = DEMc.Z(ica);
end

% include x-y-coordinates as training inputs
if addinput == 2 || addinput == 3
    Distance_test(:,end-1) = nrow(dt);
    Distance_test(:,end) = ncol(dt);
end

%normalization
Distance_test_norm = zeros(size(Distance_test));
for testnum = nrun
    for col = 1 : size(Distance_test,2)
        mean_Distance_train = mean(Distance_train{nsector,testnum}(:,col));
        std_Distance_train = std(Distance_train{nsector,testnum}(:,col));
        Distance_test_norm(:,col) = (Distance_test(:,col) - mean_Distance_train) / std_Distance_train;
    end
    mean_Thickness_train = mean(Target_train{nsector,testnum});
    std_Thickness_train = std(Target_train{nsector,testnum});
    
    
    % PREDICTION
    pred{testnum} = mlpfwd(NET{nnodes,nsector,testnum},Distance_test_norm);  % prediction on fill cells
    predT{testnum} = pred{testnum}*std_Thickness_train+mean_Thickness_train; % retransform values
    results{testnum} = zeros(MASKc.size);
    li = sub2ind(size(results{testnum}),nrow(dt),ncol(dt));
    results{testnum}(li) = predT{testnum};                          % thickness raster
    
    % apply low-pass filter
    LP = 1/9*(ones(3,3));
    results_filt{testnum} = filter2(LP,results{testnum});
    results_filt{testnum}(results{testnum}==0)=0;
    
    % compute volume
    V_pred{testnum} = sum(predT{testnum}(predT{testnum}>0))*DEMc.cellsize*DEMc.cellsize;
    [X,Y] = getcoordinates(DEMc);
    Thickness{testnum} = GRIDobj(X,Y,results_filt{testnum});        % results stored as GRIDobj in cell array 'Thickness'
    Bedrock{testnum} = GRIDobj(X,Y,DEMc.Z - results_filt{testnum}); % results converted to bedrock elevation stored as GRIDobj in cell array 'Bedrock'
end

T = zeros(size(MASKc.Z,1),size(MASKc.Z,2),nrun(end));
for i = 1 : nrun(end)
    T(:,:,i) = Thickness{i}.Z;
end

B = zeros(size(MASKc.Z,1),size(MASKc.Z,2),nrun(end));
for i = 1 : nrun(end)
    B(:,:,i) = Bedrock{i}.Z;
end

% calculate mean bedrock elevation from all predictions
meanB = mean(B,3);
stdB = std(B,0,3);
mBedrock = MASKc;
mBedrock.Z = meanB;

% load real surface
db = DEM;
db.Z = double(db.Z);
dbc = db;
dbc.Z = mBedrock.Z;
Depth_corr = db - dbc;
Depth_corr.Z(Depth_corr.Z<0) = 0;
Bed_corr = db - Depth_corr;
Depth_corr.Z(Depth_corr.Z<=0) = NaN;

% calculate mean thickness from all predictions
meanT = mean(T,3);
stdT = std(T,0,3);
mThickness = MASKc;
mThickness.Z = meanT;

close all
totaltime = toc/3600;                                               % total processing time in hours

% write geotiffs
GRIDobj2geotiff(Bed_corr,['.\',num2str(outdirection),'\Bedrock.tif']);
GRIDobj2geotiff(Depth_corr,['.\',num2str(outdirection),'\Thickness.tif']);

H = Depth_corr;
varargout{1} = Bed_corr;
varargout{2} = mRESULTSv;
varargout{3} = stdT;
