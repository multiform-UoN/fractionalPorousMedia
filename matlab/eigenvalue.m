
b=-1;


y=chebfun('y',[-b.^2/4+1e-5,100]);
disc = sqrt(b.^2 + 4*y);
mu1 = -b + disc;
mu2 = - mu1 - 2*b;
f= mu1.*exp(mu1/2)-mu2.*exp(mu2/2);
l = roots(f);

y= chebfun('y',[-100,-b.^2/4-1e-5]);
disc = sqrt(-b.^2 - 4*y);
f = -b*sin(disc/2) + disc.*cos(disc/2);
l = max([roots(f);l]);

x=chebfun('x',[0,1]);
if (b.^2+4*l>0)
    discl = sqrt(b.^2 + 4*l);
    u = exp((-b+discl)*x/2) - exp((-b-discl)*x/2);
else
    discl = sqrt(-b.^2 - 4*l);
    u = exp(-x*b/2).*sin(x*disc(l)/2);
end

u = u/norm(u);


L=chebop(@(u) diff(u,2)+b*diff(u),[0,1]);
L.lbc='dirichlet';
L.rbc='neumann';

[v,ll]=eigs(L,1);
v = v/norm(v);

uu = exp(x*b/2).*sin(x*discl/2);
uu = uu/norm(uu);

disp([l,ll])

norm(u-v)