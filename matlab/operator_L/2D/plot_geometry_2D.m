function []=plot_geometry_2D(x,y,L_x,L_y,n_gauss,L_el_x,L_el_y,x_gauss,y_gauss)

% Plot of geometry

n_el_x=length(x)-1;
n_el_y=length(y)-1;

figure('Color',[1 1 1])
axes('FontSize',14)

for n=1:n_el_x+1
    plot([x(n),x(n)],[y(1),y(n_el_y+1)],'k','LineWidth',2)
    hold on
end

for n=1:n_el_y+1
    plot([x(1),x(n_el_x+1)],[y(n),y(n)],'k','LineWidth',2)
end

n=1;
for i=1:length(y)
    for j=1:length(x)
        plot(x(j),y(end-i+1),'bo','LineWidth',3)
        text(x(j)+L_el_x/10,y(end-i+1)-L_el_y/10,num2str(n),'Color','b')
        n=n+1;
    end
end

for i=1:n_el_y
    for j=1:n_el_x
        text(x(1)+L_el_x/2+L_el_x*(j-1),y(1)+L_el_y*(n_el_y-1/2)-L_el_y*(i-1),num2str((i-1)*n_el_x+j),'Color','k')
    end
end

for i=1:n_el_y
    for j=1:n_el_x
        for n=1:n_gauss
            plot(x(1)+L_el_x*(j-1)+x_gauss(n),y(1)+L_el_y*(i-1)+y_gauss(n),'ro','LineWidth',3)
            text(x(1)+L_el_x*(j-1)+x_gauss(n)+(L_el_x-max(x_gauss))/2,y(1)+L_el_y*(i-1)+y_gauss(n)+(L_el_y-max(y_gauss))/2,num2str(n),'Color','r')
        end
    end
end

for i=1:n_el_y
    for j=1:n_el_x
        text(x(1)+L_el_x/2+L_el_x*(j-1)-L_el_x/2+L_el_x/10,y(1)+L_el_y*(n_el_y-1/2)-L_el_y*(i-1)-L_el_y/2+L_el_y/10,'1','Color','c')
        text(x(1)+L_el_x/2+L_el_x*(j-1)+L_el_x/2-L_el_x/10,y(1)+L_el_y*(n_el_y-1/2)-L_el_y*(i-1)-L_el_y/2+L_el_y/10,'2','Color','c')
        text(x(1)+L_el_x/2+L_el_x*(j-1)+L_el_x/2-L_el_x/10,y(1)+L_el_y*(n_el_y-1/2)-L_el_y*(i-1)+L_el_y/2-L_el_y/10,'3','Color','c')
        text(x(1)+L_el_x/2+L_el_x*(j-1)-L_el_x/2+L_el_x/10,y(1)+L_el_y*(n_el_y-1/2)-L_el_y*(i-1)+L_el_y/2-L_el_y/10,'4','Color','c')
    end
end

hold off
title('Geometry','FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
grid off
xlim([x(1)-L_x/10,x(end)+L_x/10])
ylim([y(1)-L_y/10,y(end)+L_y/10])

end