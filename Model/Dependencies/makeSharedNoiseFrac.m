function [noiseamps,noisefreqs,samenoise] = makeSharedNoiseFrac(noisefrac,noiseamp,noisefreq)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% noisefrac is the fraction of the total noise that is 

totalnoisevar = noiseamp.^2;
noisevars = [noisefrac.*totalnoisevar (1-noisefrac).*totalnoisevar];
noiseamps = sqrt(noisevars);

noisefreqs = [noisefreq noisefreq];
samenoise = [true false];


%%
% xtot = OUNoise(noisefreq,noiseamp,1000,0.1,1,1);
% x1 = OUNoise(noisefreq,noiseamps(1),1000,0.1,1,1);
% x2 = OUNoise(noisefreq,noiseamps(2),1000,0.1,1,1);
% 
% %%
% figure
% subplot(2,2,1)
% hist(xtot)
% subplot(2,2,3)
% hist(x1+x2)
end

