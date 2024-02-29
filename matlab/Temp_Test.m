
close all ; clear all 

% Spatial domain and mesh
a = 0 ; b = 1 ; N = 50 ; Dx = (b-a)/(N-1) ; 
u0 = ones(N,1) ; 

% Time domain and mesh 
t0 = 0 ; T = 4 ; h = 0.01 ; 

% Parameters
al = 0.8 ;
phi = 0.5*ones(N,1) ; 
beta = 0.5*ones(N,1) ; 
c_diff = 0.01 ; c_advec = 0.0 ; 
e = eye(N,1) ; F_Fun = @(t) 0*e ; 
LL = c_diff*gallery('tridiag',N,1,-2,1)/Dx^2 + c_advec*gallery('tridiag',N,-1,0,1)/2/Dx;
[t, u] = Basset(al,phi,beta,LL,F_Fun,t0,T,u0,h) ;

u_min = min(min(u)) ; u_max = max(max(u)) ;

%% Plot of the solution
figure(1) 
plot(Dx*(0:N-1),u(:,end)) 
xlabel('x')
axis([a,b,u_min,u_max]) ;
title(sprintf('Solution at T = %f',T))

%% Polot of the solution over the time
figure(2)
u_min = min(min(u)) ; u_max = max(max(u)) ;
K = size(u,2) ; 
F(K) = struct('cdata',[],'colormap',[]);
for k = 1 : 10 : K
    plot(Dx*(0:N-1),u(:,k))
    text(0.1,0.9,sprintf('t=%4.2f',t(k)), ...
        'Units','Normalized','FontSize',13) ;
    xlabel('x') ; ylabel('u(x,t)') ;
    axis([a,b,u_min,u_max]) ;
    %drawnow ;
    F(k) = getframe(gcf);
end

save("./results/out.mat","t","u")
