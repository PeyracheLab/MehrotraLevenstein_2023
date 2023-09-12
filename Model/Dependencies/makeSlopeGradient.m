function [grad] = makeSlopeGradient(N_neurons,range,midpoint)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

grad = fliplr(linspace(1-range,1+range,N_neurons))';
grad = midpoint.*grad;


end

