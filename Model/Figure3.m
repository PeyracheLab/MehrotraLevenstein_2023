savepath = '/Users/dl2820/Project Repos/UPDOWNGrad/LabNotebook/Figures_20221102';

%% Basic parms: h

simtime = 15000;
simtime = 20000;
dt = 1;

basicparms.N_neurons = 1;
%parms.W = 6;
basicparms.beta = -0.75;
basicparms.tau_r = 1;
basicparms.tau_a = 20;

basicparms.A0 = 0.3;
basicparms.Ak = -30;


basicparms.noiseamp =0.3;
basicparms.noisefreq = 0.05;

basicparms.W = 6.05;
basicparms.I_in = -3.2;
%% h current: Example
clear parms Y_sol r a 

N_neurons = 100;

%I_in = -2.4;
gradmag = -0.5;

width_line = 0.05;
width = width_line.*N_neurons; %Convert to unitary units

noisefrac = [0.25];



parms = basicparms;
parms.N_neurons = N_neurons;
parms.W = makeLineWeights(N_neurons,width,basicparms.W);
parms.beta = makeSlopeGradient(N_neurons,gradmag,basicparms.beta);
[parms.noiseamp,parms.noisefreq,parms.samenoise] = ...
    makeSharedNoiseFrac(noisefrac,basicparms.noiseamp,basicparms.noisefreq);

[ T, Y_sol ] = WCadapt_run(simtime,dt,parms);
r = Y_sol(:,1:parms.N_neurons);
a = Y_sol(:,parms.N_neurons+1:end);

%%
[UDints_mean] = DetectUD(r,'multipop','mean');

whichtrans = {'UD','DU'};
range = {[50 30],[30 120]};
for xx = 1:2
    peth.(whichtrans{xx}) = PETH(r,UDints_mean.downints(:,xx),'win',range{xx});
	[slope.(whichtrans{xx}), crossings.(whichtrans{xx})] = CalculateUDSpeed_fromPETH(peth.(whichtrans{xx}),whichtrans{xx});
    peakmag.(whichtrans{xx}) = CalculateUPPeak(peth.(whichtrans{xx}),r,UDints_mean);

end

%%
%rate_cmap = makeColorMap([1 1 1],[0 0 0],[1 0 0])
figure
xwin = [1800 2700];
subplot(4,2,1)
    imagesc(T,[0 1],r')
    hold on
    plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    %xlim([5000 6000])
    %clim([0 1])
   % crameri('lajolla')
   colormap(gca,inferno)
   % colormap(rate_cmap)
    ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
subplot(8,2,5)
    imagesc(T,[0 1],-a'.*parms.beta)
    hold on
    %plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    colorbar
    ColorbarWithAxis([0 1],'I_h')
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
    
for xx = 1:2
subplot(4,4,2+xx)
    imagesc(peth.(whichtrans{xx})')
    hold on
    plot(range{xx}(1).*[1 1],ylim(gca),'w--')
    caxis([0 1])
    colormap(gca,inferno)
    axis xy
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    %xlim([1000 2000])
    set(gca,'xtick',[]);set(gca,'ytick',[])
    title(['Slope: ',num2str(round(slope.(whichtrans{xx}),1))])
end

ordermap = makeColorMap([1 0.5 0],[0.5 0 0],10);
for xx = 1:2
subplot(4,4,6+xx)
    
    plot(peth.(whichtrans{xx})(:,1:10:end),'linewidth',1)
    colororder(gca,ordermap)
    hold on
    plot(range{xx}(1).*[1 1],ylim(gca),'k--')
 
end

for xx = 1:2
subplot(4,4,10+xx)
    
    ScatterWithLinFit(crossings.(whichtrans{xx}),peakmag.(whichtrans{xx}),'color','k','corrtype','spearman')

    xlabel([(whichtrans{xx}),' Onset Time'])
    ylabel([(whichtrans{xx}),' Peak Magnitude'])
 
end

NiceSave('Ihgrad_example',savepath,'AdWC_Line')





%% basic parms: others

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
basicparms.I_in = -2.35;
%% I: Example
clear parms Y_sol r a 

N_neurons = 100;

I_in = -2.35;
gradmag = 0.085;

width_line = 0.075;
width = width_line.*N_neurons; %Convert to unitary units

noisefrac = [0.25];



parms = basicparms;
parms.N_neurons = N_neurons;
parms.W = makeLineWeights(N_neurons,width,basicparms.W);
parms.I_in = makeSlopeGradient(N_neurons,gradmag,I_in);
[parms.noiseamp,parms.noisefreq,parms.samenoise] = ...
    makeSharedNoiseFrac(noisefrac,basicparms.noiseamp,basicparms.noisefreq);

[ T, Y_sol ] = WCadapt_run(simtime,dt,parms);
r = Y_sol(:,1:parms.N_neurons);
a = Y_sol(:,parms.N_neurons+1:end);

%%
[UDints_mean] = DetectUD(r,'multipop','mean');

whichtrans = {'UD','DU'};
range = {[50 30],[30 50]};
for xx = 1:2
    peth.(whichtrans{xx}) = PETH(r,UDints_mean.downints(:,xx),'win',range{xx});
	slope.(whichtrans{xx}) = CalculateUDSpeed_fromPETH(peth.(whichtrans{xx}),whichtrans{xx});
    peakmag.(whichtrans{xx}) = CalculateUPPeak(peth.(whichtrans{xx}),r,UDints_mean);
end

%%
%rate_cmap = makeColorMap([1 1 1],[0 0 0],[1 0 0])
figure
xwin = [1800 2700];
subplot(4,2,1)
    imagesc(T,[0 1],r')
    hold on
    plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    %xlim([5000 6000])
    %clim([0 1])
   % crameri('lajolla')
   colormap(gca,inferno)
   % colormap(rate_cmap)
    ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
subplot(8,2,5)
    imagesc(T,[0 1],a'.*parms.beta)
    hold on
    %plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    colorbar
    ColorbarWithAxis([0 1],'I_h')
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
    
for xx = 1:2
subplot(4,4,2+xx)
    imagesc(peth.(whichtrans{xx})')
    hold on
    plot(range{xx}(1).*[1 1],ylim(gca),'w--')
    caxis([0 1])
    colormap(gca,inferno)
    axis xy
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    %xlim([1000 2000])
    set(gca,'xtick',[]);set(gca,'ytick',[])
    title(['Slope: ',num2str(round(slope.(whichtrans{xx}),1))])
end

for xx = 1:2
subplot(4,4,6+xx)
    plot(peth.(whichtrans{xx})(:,1:10:end))
    colormap(gca,inferno)
end

for xx = 1:2
subplot(4,4,10+xx)
    
    ScatterWithLinFit(crossings.(whichtrans{xx}),peakmag.(whichtrans{xx}),'color','k','corrtype','spearman')

    xlabel([(whichtrans{xx}),' Onset Time'])
    ylabel([(whichtrans{xx}),' Peak Magnitude'])
 
end

NiceSave('Igrad_example',savepath,'AdWC_Line')



%%
%% W: Example
clear parms Y_sol r a 

N_neurons = 100;

gradmag = -0.075;

width_line = 0.1;
width = width_line.*N_neurons; %Convert to unitary units

noisefrac = [0.5];



parms = basicparms;
parms.N_neurons = N_neurons;
W = makeLineWeights(N_neurons,width,basicparms.W);
parms.W = makeSlopeGradient(N_neurons,gradmag,W);
[parms.noiseamp,parms.noisefreq,parms.samenoise] = ...
    makeSharedNoiseFrac(noisefrac,basicparms.noiseamp,basicparms.noisefreq);

[ T, Y_sol ] = WCadapt_run(simtime,dt,parms);
r = Y_sol(:,1:parms.N_neurons);
a = Y_sol(:,parms.N_neurons+1:end);

%%
[UDints_mean] = DetectUD(r,'multipop','mean');

whichtrans = {'UD','DU'};
range = {[50 30],[30 50]};
for xx = 1:2
    peth.(whichtrans{xx}) = PETH(r,UDints_mean.downints(:,xx),'win',range{xx});
	slope.(whichtrans{xx}) = CalculateUDSpeed_fromPETH(peth.(whichtrans{xx}),whichtrans{xx});
    peakmag.(whichtrans{xx}) = CalculateUPPeak(peth.(whichtrans{xx}),r,UDints_mean);
end

%%
%rate_cmap = makeColorMap([1 1 1],[0 0 0],[1 0 0])
figure
xwin = [1800 2700];
subplot(4,2,1)
    imagesc(T,[0 1],r')
    hold on
    plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    %xlim([5000 6000])
    %clim([0 1])
   % crameri('lajolla')
   colormap(gca,inferno)
   % colormap(rate_cmap)
    ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
subplot(8,2,5)
    imagesc(T,[0 1],a'.*parms.beta)
    hold on
    %plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    colorbar
    ColorbarWithAxis([0 1],'I_h')
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
    
for xx = 1:2
subplot(4,4,2+xx)
    imagesc(peth.(whichtrans{xx})')
    hold on
    plot(range{xx}(1).*[1 1],ylim(gca),'w--')
    caxis([0 1])
    colormap(gca,inferno)
    axis xy
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    %xlim([1000 2000])
    set(gca,'xtick',[]);set(gca,'ytick',[])
    title(['Slope: ',num2str(round(slope.(whichtrans{xx}),1))])
end

for xx = 1:2
subplot(4,4,6+xx)
    plot(peth.(whichtrans{xx})(:,1:10:end))
    colormap(gca,inferno)
end

for xx = 1:2
subplot(4,4,10+xx)
    
    ScatterWithLinFit(crossings.(whichtrans{xx}),peakmag.(whichtrans{xx}),'color','k','corrtype','spearman')

    xlabel([(whichtrans{xx}),' Onset Time'])
    ylabel([(whichtrans{xx}),' Peak Magnitude'])
 
end

NiceSave('Wgrad_example',savepath,'AdWC_Line')



%%
%% b: Example
clear parms Y_sol r a 

N_neurons = 100;

gradmag = 0.15;

width_line = 0.1;
width = width_line.*N_neurons; %Convert to unitary units

noisefrac = [0.5];



parms = basicparms;
parms.N_neurons = N_neurons;
parms.W = makeLineWeights(N_neurons,width,basicparms.W);
parms.beta = makeSlopeGradient(N_neurons,gradmag,basicparms.beta);
[parms.noiseamp,parms.noisefreq,parms.samenoise] = ...
    makeSharedNoiseFrac(noisefrac,basicparms.noiseamp,basicparms.noisefreq);

[ T, Y_sol ] = WCadapt_run(simtime,dt,parms);
r = Y_sol(:,1:parms.N_neurons);
a = Y_sol(:,parms.N_neurons+1:end);

%%
[UDints_mean] = DetectUD(r,'multipop','mean');

whichtrans = {'UD','DU'};
range = {[50 30],[30 50]};
for xx = 1:2
    peth.(whichtrans{xx}) = PETH(r,UDints_mean.downints(:,xx),'win',range{xx});
	slope.(whichtrans{xx}) = CalculateUDSpeed_fromPETH(peth.(whichtrans{xx}),whichtrans{xx});
    peakmag.(whichtrans{xx}) = CalculateUPPeak(peth.(whichtrans{xx}),r,UDints_mean);
end

%%
%rate_cmap = makeColorMap([1 1 1],[0 0 0],[1 0 0])
figure
xwin = [1800 2700];
subplot(4,2,1)
    imagesc(T,[0 1],r')
    hold on
    plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    %xlim([5000 6000])
    %clim([0 1])
   % crameri('lajolla')
   colormap(gca,inferno)
   % colormap(rate_cmap)
    ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
subplot(8,2,5)
    imagesc(T,[0 1],a'.*parms.beta)
    hold on
    %plot(T,0.5.*mean(r,2)+1,'k','linewidth',1)
    axis xy
    axis tight
    colorbar
    ColorbarWithAxis([0 1],'I_h')
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    xlim(xwin)
    set(gca,'xtick',[]);set(gca,'ytick',[])
    box off
    
    
for xx = 1:2
subplot(4,4,2+xx)
    imagesc(peth.(whichtrans{xx})')
    hold on
    plot(range{xx}(1).*[1 1],ylim(gca),'w--')
    caxis([0 1])
    colormap(gca,inferno)
    axis xy
    %xlim([5000 6000])
    %ColorbarWithAxis([0 1],'R')
    %xlabel('t');ylabel('x')
    %xlim([1000 2000])
    set(gca,'xtick',[]);set(gca,'ytick',[])
    title(['Slope: ',num2str(round(slope.(whichtrans{xx}),1))])
end

for xx = 1:2
subplot(4,4,6+xx)
    plot(peth.(whichtrans{xx})(:,1:10:end))
    colormap(gca,inferno)
end

for xx = 1:2
subplot(4,4,10+xx)
    
    ScatterWithLinFit(crossings.(whichtrans{xx}),peakmag.(whichtrans{xx}),'color','k','corrtype','spearman')

    xlabel([(whichtrans{xx}),' Onset Time'])
    ylabel([(whichtrans{xx}),' Peak Magnitude'])
 
end

NiceSave('bgrad_example',savepath,'AdWC_Line')



