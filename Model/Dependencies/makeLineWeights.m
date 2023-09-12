function [WeightMat] = makeLineWeights(N_neurons,width,W)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Nidx = 1:N_neurons;

if length(W)==1
    W = W.*ones(size(Nidx));
end

for xx = Nidx
    WeightMat(xx,:)=Gauss(Nidx,xx,width)./sum(Gauss(Nidx,xx,width));
    WeightMat(xx,:) = WeightMat(xx,:) * W(xx);
end


end

