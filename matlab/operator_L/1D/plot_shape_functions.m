function []=plot_shape_functions(csi,N)

% Plot of shape functions
[~,c]=size(N);
col=['r','b','g'];
for n=1:c
    str_leg(n) = java.lang.String(['N_',num2str(n),' (\xi)']);
    leg = cell(str_leg);
end
figure('Color',[1 1 1])
axes('FontSize',14)
for n=1:c
    plot(csi,N(:,n),col(n),'LineWidth',3)
    hold on
end
plot(csi,zeros(1,length(csi)),'k','LineWidth',3)
hold off
title('Shape functions','FontSize',14)
xlabel('\xi','FontSize',14)
ylabel('N','FontSize',14)
legend(leg)
grid on
grid minor
xlim([-1.1,+1.1])
ylim([-0.5,+1.5])

end