function [A_ff,A_fp,A_pf,A_pp]=constrain_matrix(A,dof_constrained)

% Constrain a matrix
N=length(A);
p=dof_constrained;
f_aus=1:N;
p_aus=zeros(1,N);
p_aus(p)=p;
f=f_aus-p_aus;
f=find(f);

A_ff=A(f,f);
A_fp=A(f,p);
A_pf=A(p,f);
A_pp=A(p,p);

end