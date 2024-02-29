function [csi,w]=Gauss_parameters(n)

% Parameters for Gauss integration rule in [-1,+1]
switch n
    case 1 % Gauss with 1 nodes
        csi=[0];
        w=[2];
    case 2 % Gauss with 2 nodes        
        csi=[-1/sqrt(3),+1/sqrt(3)];
        w=[1,1];
    case 3 % Gauss with 3 nodes
        csi=[-sqrt(3/5),0,+sqrt(3/5)];
        w=[5/9,8/9,5/9];
    case 4 % Gauss with 4 nodes
        csi=[-1/35*sqrt(525+70*sqrt(30)),-1/35*sqrt(525-70*sqrt(30)),...
             +1/35*sqrt(525-70*sqrt(30)),+1/35*sqrt(525+70*sqrt(30))];
        w=[1/36*(18-sqrt(30)),1/36*(18+sqrt(30)),...
           1/36*(18+sqrt(30)),1/36*(18-sqrt(30))];
end

end

