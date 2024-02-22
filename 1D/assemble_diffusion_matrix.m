function [K]=assemble_diffusion_matrix(el,dof,n_el,dof_el,A)

% Assemblage of diffusion matrix
K=zeros(dof,dof);
for n=1:n_el
    for i=1:dof_el
        for j=1:dof_el
            K(A(n,i),A(n,j))=K(A(n,i),A(n,j))+el(n).K(i,j);
        end
    end
end

end

