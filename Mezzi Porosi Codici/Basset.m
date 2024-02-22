function [t, u] = Basset(al,phi,beta,LL,F_fun,t0,T,u0,h)
% al : fractional order
% phi, beta : vector of the same size of u0
% LL : discretization of the diffusione-advection-reaction operator
% F_fun : forcing term
% t0, T : integration time window
% u0 : initial solution
% h : time step
% ================================================================

% Number of points in which to evaluate the solution and the weights
r = 64 ; 
N = ceil((T-t0)/h) ;
Nr = ceil((N+1)/r)*r ;
Qr = ceil(log2((Nr)/r)) - 1 ;
NNr = 2^(Qr+1)*r ;


% Evaluation of time grid
t = t0 + (0:N)*h ;

% Evaluation of cofficients of the rectangular product integration
nvett = 0 : NNr+1 ; 
nal = nvett.^(1-al) ; 
bn = ( nal(2:end) - nal(1:end-1) ) / gamma(2-al) ; 


A_h = diag(phi) + h^(1-al)*bn(1)*diag(beta) - h*LL ; 
[LA,UA,PA] = lu(A_h) ;

Probl.al = al ; 
Probl.phi = phi(:) ; 
Probl.beta = beta(:) ; 
Probl.u0 = u0(:) ;
Probl.LL = LL ; 
Probl.LA = LA ; Probl.UA = UA ; Probl.PA = PA ;
Probl.F_fun = F_fun ; 
Probl.size = length(u0) ; 

% Preallocation of some variables
u = zeros(Probl.size,N+1) ;
u_mem_int = zeros(Probl.size,1) ; 
F_mem_int = zeros(Probl.size,1) ; 
zn = zeros(Probl.size,NNr+1) ;

METH.bn = bn ; METH.h = h ;

% Initializing solution and proces of computation
u(:,1) = u0(:,1) ;
[u, u_mem_int, F_mem_int] = Triangolo(1, r-1, t, u, u_mem_int, F_mem_int, zn, N, METH, Probl ) ;

% Main process of computation by means of the FFT algorithm
ff = zeros(1,2.^(Qr+2)) ; ff(1:2) = [0 2] ; card_ff = 2 ;
nx0 = 0 ; ny0 = 0 ;
for qr = 0 : Qr
    L = 2^qr ; 
    [u, u_mem_int, F_mem_int] = DisegnaBlocchi(L, ff, r, Nr, nx0+L*r, ny0, t, ...
        u, u_mem_int, F_mem_int, zn, N, METH, Probl ) ;
    ff(1:2*card_ff) = [ff(1:card_ff) ff(1:card_ff)] ; 
    card_ff = 2*card_ff ; 
    ff(card_ff) = 4*L ; 
end

% Evaluation solution in TFINAL when TFINAL is not in the mesh
if T < t(N+1)
    c = (T - t(N))/h ;
    t(N+1) = tfinal ;
    u(:,N+1) = (1-c)*u(:,N) + c*u(:,N+1) ;
end

t = t(1:N+1) ; u = u(:,1:N+1) ;

end


% =========================================================================
% =========================================================================
% r : dimension of the basic square or triangle
% L : factor of resizing of the squares
function [u, u_mem_int, F_mem_int] = DisegnaBlocchi(L, ff, r, Nr, nx0, ny0, t, ...
    u, u_mem_int, F_mem_int, zn, N , METH, Probl)

nxi = nx0 ; nxf = nx0 + L*r - 1 ;
nyi = ny0 ; nyf = ny0 + L*r - 1 ;
is = 1 ;
s_nxi(is) = nxi ; s_nxf(is) = nxf ; s_nyi(is) = nyi ; s_nyf(is) = nyf ;

i_triangolo = 0 ; stop = 0 ;
while ~stop
    
    stop = nxi+r-1 == nx0+L*r-1 | (nxi+r-1>=Nr-1) ;
    
    zn = Quadrato(nxi, nxf, nyi, nyf, u, zn, METH, Probl) ;
    
    [u, u_mem_int, F_mem_int] = Triangolo(nxi, nxi+r-1, t, u, u_mem_int, F_mem_int, zn, N, METH, Probl) ;
    i_triangolo = i_triangolo + 1 ;
    
    if ~stop
        if nxi+r-1 == nxf   % Il triangolo finisce dove finisce il quadrato -> si scende di livello
            i_Delta = ff(i_triangolo) ;
            Delta = i_Delta*r ;
            nxi = s_nxf(is)+1 ; nxf = s_nxf(is)  + Delta ;
            nyi = s_nxf(is) - Delta +1; nyf = s_nxf(is)  ;
            s_nxi(is) = nxi ; s_nxf(is) = nxf ; s_nyi(is) = nyi ; s_nyf(is) = nyf ;
        else % Il triangolo finisce prima del quadrato -> si fa un quadrato accanto
            nxi = nxi + r ; nxf = nxi + r - 1 ; nyi = nyf + 1 ; nyf = nyf + r  ;
            is = is + 1 ;
            s_nxi(is) = nxi ; s_nxf(is) = nxf ; s_nyi(is) = nyi ; s_nyf(is) = nyf ;
        end
    end
    
end

end


% =========================================================================
% =========================================================================
function zn = Quadrato(nxi, nxf, nyi, nyf, u, zn, METH, Probl)

coef_beg = nxi-nyf ; coef_end = nxf-nyi+1 ;
funz_beg = nyi+1 ; funz_end = nyf+1 ;

% Evaluation convolution segments 

vett_coef = METH.bn(coef_beg:coef_end) ;
if nyi == 0 % Evaluation of the lowest square
    vett_funz = [zeros(Probl.size,1) , ...
        u(:,funz_beg+1:funz_end) , ...
        zeros(Probl.size,funz_end-funz_beg+1) ] ;
else % Evaluation of any square but not the lowest
    vett_funz = [ u(:,funz_beg:funz_end) , zeros(Probl.size,funz_end-funz_beg+1) ] ;
end
zzn = real(FastConv(vett_coef,vett_funz)) ;
zn(:,nxi+1:nxf+1) = zn(:,nxi+1:nxf+1) + zzn(:,nxf-nyf+1:end) ;
end




% =========================================================================
% =========================================================================
function [u, u_mem_int, F_mem_int] = Triangolo(nxi, nxf, t, u, u_mem_int, F_mem_int, zn, N, METH, Probl)

al = Probl.al ;
h = METH.h ; 

for n = nxi : min(N,nxf)
    
    % Evaluation of the convolution term
    temp = zn(:,n+1) ; 
    for j = nxi : n-1
        temp = temp + METH.bn(n-j+1)*u(:,j+1) ;
    end
    
    ttemp = 0 ;
    for j = 1 : n-1
        ttemp = ttemp + METH.bn(n-j+1)*u(:,j+1) ;
    end 

    F_mem_int = F_mem_int + Probl.F_fun(t(n+1)) ;
    
    F_n = (Probl.phi + Probl.beta*t(n+1)^(1-al)/gamma(2-al)).*Probl.u0 ...
        - h^(1-al)*Probl.beta.*temp ...
        + h*Probl.LL*u_mem_int + h*F_mem_int;
    
    u(:,n+1) = Probl.UA\(Probl.LA\(Probl.PA*F_n)) ;
    u_mem_int = u_mem_int + u(:,n+1) ; 
    

end

end


% =========================================================================
% =========================================================================
function z = FastConv(x, y)

Lx = length(x) ; Ly = size(y,2) ; problem_size = size(y,1) ;
if Lx ~= Ly
    disp('Warning: dimensions in FastConv do not agree') ;
end
r = Lx ;


z = zeros(problem_size,r) ;
X = fft(x,r) ;
for i = 1 : problem_size
    Y = fft(y(i,:),r) ;
    Z = X.*Y ;
    z(i,:) = ifft(Z,r) ;
end

end






