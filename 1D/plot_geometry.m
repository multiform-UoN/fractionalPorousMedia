function []=plot_geometry(x,h,n_np,L,n_el,n_gauss,L_el,x_gauss)

% Plot of geometry
figure('Color',[1 1 1])
axes('FontSize',14)
plot(x,zeros(1,length(x)),'k','LineWidth',3)
hold on
for n=1:n_np
    plot([x(1)+h*(n-1),x(1)+h*(n-1)],[-L/20,+L/20],'k','LineWidth',2)
    text(x(1)+h*(n-1)-h/15,-L/10,num2str(n))
end
for i=1:n_el
    for n=1:n_gauss
        plot([x(1)+(i-1)*L_el+x_gauss(n),x(1)+(i-1)*L_el+x_gauss(n)],...
             [-L/30,+L/30],'r','LineWidth',2)
    end
end
hold off
title('Geometry','FontSize',14)
xlabel('x [m]','FontSize',14)
ylabel('y [m]','FontSize',14)
grid on
grid minor
xlim([x(1)-L/10,x(end)+L/10])
ylim([-L,L])

end