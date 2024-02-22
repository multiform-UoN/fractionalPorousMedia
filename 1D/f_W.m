function [W] = f_W(csi,beta)

% Test functions
W1=1/2*(1-csi)-3/4*beta*(1-csi.^2);
W2=1/2*(1+csi)+3/4*beta*(1-csi.^2);
W=[W1',W2'];

end

