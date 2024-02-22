function [f]=element_load_vector(s,dof_el,n_gauss,N,W,w,J)

% Element load vector
f=zeros(dof_el,1);
for i=1:dof_el
    for j=1:dof_el
        for nn=1:n_gauss
            f(i,1)=f(i,1)+(W(i,nn)*W(j,nn)*s(j))*w(nn);
        end
    end
    f(i,1)=(f(i,1))*J;
end

end

