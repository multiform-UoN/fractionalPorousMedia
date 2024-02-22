function [M]=element_mass_matrix_2D(dof_el,gauss,J)

% Element mass matrix
n_gauss=length(gauss);
M=zeros(dof_el,dof_el);
for i=1:dof_el
    for j=1:dof_el
        for n=1:n_gauss
            M(i,j)=M(i,j)+(gauss(n).W(i).W*gauss(n).N(j).N)*gauss(n).w;
        end
        M(i,j)=M(i,j)*J;
    end
end

end