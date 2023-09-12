function [peth] = PETH(rate,eventtimes,varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%
p = inputParser;
addParameter(p,'win',[100 100]);
addParameter(p,'showfig',false);
parse(p,varargin{:})
win = p.Results.win;
SHOWFIG = p.Results.showfig;
%note to self, this is all terrible because assuming timestamps.... 0_o
%%

[totalduration,numunits] = size(rate);
numevents = length(eventtimes);


rate = [nan(win(1),numunits); rate; nan(win(2),numunits)];
eventwins = eventtimes + [0 sum(win)];

peth = zeros(sum(win)+1,numunits,numevents);
for ee = 1:numevents
    peth(:,:,ee) = rate(eventwins(ee,1):eventwins(ee,2),:);
end
peth = nanmean(peth,3);

%%
if SHOWFIG
    figure
    imagesc(peth')
end
end

