function [UDspeed,crossings] = CalculateUDSpeed_fromPETH(peth,ud)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%%
SHOWFIG = false;
thresh = 0.5;
mincrossings = 5;
%%

peth = peth-min(peth(:));
peth = peth./max(peth(:));
if strcmp(ud,'UD')
    %Flip everything for up->DOWN
    peth = -peth;
    thresh = -thresh;
elseif strcmp(ud,'DU')
else
    error('input ud should be "DU" or "UD"')
end

[numt,numchannels] = size(peth);
popnum = 1:numchannels;
%Get threshold crossings
%(if strcmp(ud,'UD'), crossings down. if strcmp(ud,'UD'), crossings up.)
crossings = nan(1,numchannels);
for cc = 1:numchannels
    thiscrossings = find(peth(1:end-1,cc)<=thresh & peth(2:end,cc)>thresh);
    if isempty(thiscrossings)
        continue
    elseif length(thiscrossings)>1
        %keyboard
        %Find the crossing closest to the adjacent crossing (hack)
        if cc==1 | isnan(crossings(cc-1))
            thiscrossings = mean(thiscrossings);
        else
            [~,closestcrossing] = min(abs(thiscrossings-crossings(cc-1))); 
            thiscrossings = thiscrossings(closestcrossing);
        end
    end
    crossings(cc) = thiscrossings;
             
end
%Fit line

%%
if sum(~isnan(crossings))<mincrossings
    display('Not enough crossings')
    UDspeed = nan;
    return
end

%%
popnum(isnan(crossings)) = [];
crossings(isnan(crossings)) = [];
c = polyfit(popnum,crossings,1);
UDspeed = c(1);
%UDspeed = 1./slope;



%%
if SHOWFIG
    figure
        subplot(2,2,1)
            plot(crossings,popnum,'.')
            hold on
            plot(polyval(c,popnum),popnum,'r--')
            %title(whichtrans{xx})
            xlabel('Relative time')
            ylabel('Population')
end
