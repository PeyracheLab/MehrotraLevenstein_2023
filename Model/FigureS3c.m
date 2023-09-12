function LabNotebook20211117b_WidthGradWSpeed(savepath)

%%
%savepath = '/Users/dl2820/Project Repos/UPDOWNGrad/LabNotebook/Figures_20211117b';

savefilename = fullfile(savepath,'simruns.mat');
if ~exist(savepath,'dir')
    mkdir(savepath)
end
%%
display(['Will save to ',savepath])
pc = parcluster('local');
    % sto
% % store temporary files in the 'scratch' drive on the cluster, labeled by job ID
pc.JobStorageLocation = strcat(getenv('SCRATCH'), '/', getenv('SLURM_JOB_ID'));
% % enable MATLAB to utilize the multiple cores allocated in the job script
% % SLURM_NTASKS_PER_NODE is a variable set in the job script by the flag --tasks-per-node
% % we use SLURM_NTASKS_PER_NODE - 1, because one of these tasks is the original MATLAB script itself
parpool(pc, str2num(getenv('SLURM_NTASKS_PER_NODE'))-1);

%% Basic Parameters: ExcitableUP

simtime = 20000;
dt = 1;

basicparms.N_neurons = 1;
%parms.W = 6;
basicparms.beta = 1;
basicparms.tau_r = 1;
basicparms.tau_a = 20;

basicparms.A0 = 0.5;
basicparms.Ak = 15;

basicparms.noiseamp =0.25;
basicparms.noisefreq = 0.05;

basicparms.W = 6.05;
basicparms.I_in = -2.4;
%% ExcitableUP Simulation: 1 Unit
[ T, Y_sol ] = WCadapt_run(simtime,dt,basicparms);
r = Y_sol(:,1:basicparms.N_neurons);
a = Y_sol(:,basicparms.N_neurons+1:end);

%% Figure
figure
    subplot(4,1,1)
        hold on
        plot(T,r,'k')
        plot(T,a,'b')

NiceSave('ExcitableUP',savepath,'1pop')




%% ExcitableUP Simulation: 2d - width vs frac
clear parms Y_sol r a 

N_neurons = 100;

%I_in = -2.4;
gradmag = logspace(-2.5,-1,8);

width_line = logspace(-2.25,-0.5,8);
width = width_line.*N_neurons; %Convert to unitary units

noisefrac = [0 0.25 0.5];

numsims = length(noisefrac)*length(gradmag)*length(width);
parfor nn = 1:numsims
    bz_Counter(nn,numsims,'Sim')
    [ff,gg,ww] = ind2sub([length(noisefrac),length(gradmag),length(width)],nn);
    
    thisfrac = noisefrac(ff);
    thiswidth = width(ww);
    thisgrad = gradmag(gg);

    parms_temp{nn} = basicparms;
    parms_temp{nn}.N_neurons = N_neurons;
    W = makeSlopeGradient(N_neurons,thisgrad,basicparms.W);
    parms_temp{nn}.W = makeLineWeights(N_neurons,thiswidth,W);
    [parms_temp{nn}.noiseamp,parms_temp{nn}.noisefreq,parms_temp{nn}.samenoise] = ...
        makeSharedNoiseFrac(thisfrac,basicparms.noiseamp,basicparms.noisefreq);

    [ T, Y_sol ] = WCadapt_run(simtime,dt,parms_temp{nn});
    r_temp{nn} = Y_sol(:,1:parms_temp{nn}.N_neurons);

end
%% Deal stuff back out
for nn = 1:numsims
    [ff,gg,ww] = ind2sub([length(noisefrac),length(gradmag),length(width)],nn);
    parms{ww,gg,ff} = parms_temp{nn};
    r{ww,gg,ff} = r_temp{nn};
end
clear r_temp parms_temp
clear pc
save(savefilename,'-v7.3')
disp('mat file saved')
%parms{ww,gg,ff}
%r{ww,gg,ff}
%%
for ff = 1:length(noisefrac)

figure
for gg = 1:length(gradmag)
    for ww = 1:length(width)
     
        subplot(length(gradmag),length(width),(ww-1).*length(width)+gg)
        imagesc(T,[0 1],r{ww,gg,ff}')
        hold on
        plot(T,0.5.*mean(r{ww,gg,ff},2)+1,'k','linewidth',1)
        axis xy
        axis tight
        %xlim([5000 6000])
        %ColorbarWithAxis([0 1],'R')
        %xlabel('t');ylabel('x')
        xlim([1000 1500])
        set(gca,'xtick',[]);set(gca,'ytick',[])
        box off


    end
end
NiceSave(['WidthGrad_',num2str(noisefrac(ff)),'f'],savepath,'AdWC_Line')
end

%% Detect UD
clear UDints
for nn = 1:numsims
    bz_Counter(nn,numsims,'Sim')
    [ff,gg,ww] = ind2sub([length(noisefrac),length(gradmag),length(width)],nn);
            %Add: ignore onset?
            
            [UDints_mean(ww,gg,ff)] = DetectUD(r{ww,gg,ff},'multipop','mean');

end

%% Calculate Mean Rate relative to UD transitions
clear peth_DU peth_UD
for nn = 1:numsims
    bz_Counter(nn,numsims,'Sim')
    [ff,gg,ww] = ind2sub([length(noisefrac),length(gradmag),length(width)],nn);
            if isempty(UDints_mean(ww,gg,ff).downints)
                peth_UD{ww,gg,ff} = nan;
                peth_DU{ww,gg,ff} = nan;
                continue
            end
            peth_UD{ww,gg,ff} = PETH(r{ww,gg,ff},UDints_mean(ww,gg,ff).downints(:,1),'win',[50 30]);
            peth_DU{ww,gg,ff} = PETH(r{ww,gg,ff},UDints_mean(ww,gg,ff).downints(:,2),'win',[30 50]);

end

%% Figures

for ff = 1:length(noisefrac)
figure
for gg = 1:length(gradmag)
    for ww = 1:length(width)
     
        subplot(length(gradmag),length(width),(ww-1).*length(width)+gg)
        imagesc(peth_UD{ww,gg,ff}')
        caxis([0 1])
        axis xy
        %xlim([5000 6000])
        %ColorbarWithAxis([0 1],'R')
        %xlabel('t');ylabel('x')
        %xlim([1000 2000])
        set(gca,'xtick',[]);set(gca,'ytick',[])

    end
end
NiceSave(['UD_',num2str(noisefrac(ff)),'f'],savepath,'AdWC_Line')


figure
for gg = 1:length(gradmag)
    for ww = 1:length(width)
     
        subplot(length(gradmag),length(width),(ww-1).*length(width)+gg)
        imagesc(peth_DU{ww,gg,ff}')
        caxis([0 1])
        axis xy
        %xlim([5000 6000])
        %ColorbarWithAxis([0 1],'R')
        %xlabel('t');ylabel('x')
        %xlim([1000 2000])
        set(gca,'xtick',[]);set(gca,'ytick',[])

    end
end
NiceSave(['DU_',num2str(noisefrac(ff)),'f'],savepath,'AdWC_Line')
end


%%

for nn = 1:numsims
    [ff,gg,ww] = ind2sub([length(noisefrac),length(gradmag),length(width)],nn);
            if isempty(UDints_mean(ww,gg,ff).downints)
                slope.UD(ww,gg,ff) = nan;
                slope.DU(ww,gg,ff) = nan;
                continue
            end
            slope.UD(ww,gg,ff) = CalculateUDSpeed_fromPETH(peth_UD{ww,gg,ff},'UD');
            slope.DU(ww,gg,ff) = CalculateUDSpeed_fromPETH(peth_DU{ww,gg,ff},'DU');
end




%%

crange = [min([slope.UD(:);slope.DU(:)]),max([slope.UD(:);slope.DU(:)])];
crange= [-0.7 0.7];
whichtrans = {'UD','DU'};
figure
for ff = 1:length(noisefrac)
for xx = 1:2
    subplot(4,4,xx+(ff-1)*4)
        imagesc(log10(gradmag),log10(width_line),slope.(whichtrans{xx})(:,:,ff))
        alpha(gca,1.*~isnan(slope.(whichtrans{xx})(:,:,ff)))
        
        colorbar
        %axis xy
        ylabel('Connectivity Width')
        xlabel('Recurrence Gradient')
        title(whichtrans{xx})
        caxis(crange)
        crameri('berlin','pivot',0)
        LogScale('xy',10)
        
        
end
end
NiceSave('SlopeGdW',savepath,[])