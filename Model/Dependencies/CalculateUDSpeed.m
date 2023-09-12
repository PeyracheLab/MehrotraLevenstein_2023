function [slope] = CalculateUDSpeed(r)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%%

minjump = 100;
SHOWFIG = false;

% meanR = mean(r,2);
% 
% [UDints_mean] = DetectUD(meanR);
numpops = length(r(1,:));

for pp = 1:numpops
    bz_Counter(pp,numpops,'Pop')
    [UDints(pp)] = DetectUD(r(:,pp));
end

%%
middleunit = round(numpops/2);

whichtrans = {'UD','DU'};
clear popnum reltime c slope
for xx = 1:2
    popnum.(whichtrans{xx}) = [];
    reltime.(whichtrans{xx}) = [];
    if isempty(UDints(middleunit).downints)
        slope.(whichtrans{xx}) = [];
        continue
    end
    numrefUDs = length(UDints(middleunit).downints(:,xx)); 
    for ud = 1:numrefUDs
        %Find the closest UD to each refUD
        thisUD = UDints(middleunit).downints(ud,xx);

        %get the closest UD in each unit
        for pp = 1:numpops

            [~,closestUD] = min(abs(UDints(pp).downints(:,xx)-thisUD));
            thisrelltime = UDints(pp).downints(closestUD,xx)-thisUD;

            if abs(thisrelltime)>minjump
                continue
            end
            reltime.(whichtrans{xx}) = [reltime.(whichtrans{xx}) thisrelltime];
            popnum.(whichtrans{xx}) = [popnum.(whichtrans{xx}) pp];
        end
    end

    c.(whichtrans{xx}) = polyfit(reltime.(whichtrans{xx}),popnum.(whichtrans{xx}),1);
    slope.(whichtrans{xx}) = c.(whichtrans{xx})(1);
    
end


%%
if SHOWFIG
    figure
    for xx = 1:2
        subplot(2,2,xx)
            plot(reltime.(whichtrans{xx}),popnum.(whichtrans{xx}),'.')
            hold on
            plot(reltime.(whichtrans{xx}),polyval(c.(whichtrans{xx}),reltime.(whichtrans{xx})),'r--')
            title(whichtrans{xx})
            xlabel('Relative time')
            ylabel('Population')
    end
end
