function [peakmag] = CalculateUPPeak(peth,r,UDints_mean)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

peakmag = max(peth,[],1);

UPIDX = bz_INTtoIDX({UDints_mean.upints});
meanUPrate = mean(r(UPIDX,:),1);

peakmag = peakmag./meanUPrate;
end

