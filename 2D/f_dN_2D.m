function [dN] = f_dN_2D(csi,eta)

% 1st derivatives of shape functions
dN(1).dN=[-1/4*(1-eta);-1/4*(1-csi)];
dN(2).dN=[+1/4*(1-eta);-1/4*(1+csi)];
dN(3).dN=[+1/4*(1+eta);+1/4*(1+csi)];
dN(4).dN=[-1/4*(1+eta);+1/4*(1-csi)];

end