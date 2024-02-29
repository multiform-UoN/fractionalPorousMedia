function [dW] = f_dW(csi,beta)

% 1st derivatives of test functions
dW1=-1/2+3/2*beta*csi;
dW2=+1/2-3/2*beta*csi;
dW=[dW1',dW2'];

end
