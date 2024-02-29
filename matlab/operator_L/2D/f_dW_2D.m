function [dW] = f_dW_2D(csi,eta)

% 1st derivatives of test functions
dW(1).dW=[-1/4*(1-eta);-1/4*(1-csi)];
dW(2).dW=[+1/4*(1-eta);-1/4*(1+csi)];
dW(3).dW=[+1/4*(1+eta);+1/4*(1+csi)];
dW(4).dW=[-1/4*(1+eta);+1/4*(1-csi)];

end