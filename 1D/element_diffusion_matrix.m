function [K]=element_diffusion_matrix(v,dof_el,n_gauss,dN,dW,w,J)

% Element diffusion matrix
K=zeros(dof_el,dof_el);
for i=1:dof_el
    for j=1:dof_el
        for nn=1:n_gauss
            K(i,j)=K(i,j)+(dW(i,nn)*dN(j,nn))*w(nn);
        end
        K(i,j)=v*(K(i,j))/J;
    end
end

end

