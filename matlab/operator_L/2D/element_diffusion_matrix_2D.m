function [K]=element_diffusion_matrix_2D(v,dof_el,gauss,J)

% Element diffusion matrix
n_gauss=length(gauss);
K=zeros(dof_el,dof_el);
for i=1:dof_el
    for j=1:dof_el
        for n=1:n_gauss
            K(i,j)=K(i,j)+(gauss(n).dW(i).dW'*gauss(n).dN(j).dN)*gauss(n).w;
        end
        K(i,j)=v*K(i,j)/J;
    end
end

end