function [M]=assemble_mass_matrix(el,dof,n_el,dof_el,A)

% Assemblage of mass matrix
M=zeros(dof,dof);
for n=1:n_el
    for i=1:dof_el
        for j=1:dof_el
            M(A(n,i),A(n,j))=M(A(n,i),A(n,j))+el(n).M(i,j);
        end
    end
end

end

