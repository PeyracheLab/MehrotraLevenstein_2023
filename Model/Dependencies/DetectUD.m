function [UDints] = DetectUD(r,varargin)
%DETECTUD Summary of this function goes here
%   Detailed explanation goes here
%%
multipoptions = {false,'mean','each'};

p = inputParser;
addParameter(p,'multipop',false);
parse(p,varargin{:})
multipop = p.Results.multipop;

%%
switch multipop
    case 'mean'
        r = mean(r,2);
end
%%
[thresh,cross,ratehist] = BimodalThresh(r);

UDints = cross;
end

