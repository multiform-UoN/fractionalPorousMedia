function [W] = f_W_2D(csi,eta)

% Test functions
W(1).W=1/4*(1-csi)*(1-eta);
W(2).W=1/4*(1+csi)*(1-eta);
W(3).W=1/4*(1+csi)*(1+eta);
W(4).W=1/4*(1-csi)*(1+eta);

end