function WCadapt_plotActivationCurves(parms)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%%
r = linspace(0,1,100);

% I_tot = W*r - beta.*a + I_in + interp1(noiseT,sum(Inoise,3),t)' + ...
%     pulsefun(t,pulseparms) + rampfun(t,rampparms);
% 
% F_I = 1./(1+exp(-I_tot));
A0 = parms.A0;
Ak = parms.Ak;

Ainf = 1./(1+exp(-Ak.*(r-A0)));

figure
subplot(4,4,1)
plot(r,Ainf,'b','linewidth',2)
xlabel('r');ylabel('Ainf(r)')
end

