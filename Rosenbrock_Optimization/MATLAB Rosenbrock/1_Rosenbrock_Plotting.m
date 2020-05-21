clear all
clc
%Set variables a and b
a=1;
b=100;
%define arrays x1 and x2
x1=[-2:.1:2];
x2=x1;
%%Use meshgrid
[X,Y]=meshgrid(x1,x2);
%%Use element wise operation to create Rosenbrocks's function
f=(a-X).^2+b.*(Y-X.^2).^2;
%%Plot surface
figure('Name','Function surface plot')
surf(X,Y,f,'LineStyle','none')
view(-140,20)%To rotate the plot for a better view
xlabel('X1')
ylabel('X2')
%%Contour Plot
contourplot=figure('Name','Function Contour Plot');
set(contourplot,'Position',[10,50,2000,2000])%To increase size of figure for better visibility
contour(X,Y,f,20,'ShowText','on')
xlabel('X1')
ylabel('X2')
fprintf('It can be seen from the contour plot that there is a minima range, but it is hard to predict the exact value')
%%Cross sections at x1=1
%as seen from the result table, this occurs at X(:,31)
figure('Name','Rosenbrocks function at x1=1')
plot(Y(:,31),f(:,31))
xlabel('X2')
legend 'X1 = 1'
ylabel('Rosenbrocks function')
%%Cross sections at x1=-1 
%as seen from the result table, this occurs at X(:,11)
figure('Name','Rosenbrocks function at x1=-1')
plot(Y(:,11),f(:,11))
xlabel('X2')
ylabel('Rosenbrocks function')
legend 'X1 = - 1'
%%Cross sections at x2=1
%as seen from the result table, this occurs at Y(31,:)
figure('Name','Q2e(iii)Rosenbrocks function at x2=1')
plot(X(31,:),f(31,:))
xlabel('X1')
ylabel('Rosenbrocks function')
legend 'X2 = 1'n 
%%Plot analysis
fprintf(' The minima is clearly visible at (1,1). (-1,1) is a local minimum, but not a global minumum as there is a value of x = (1,1) where function value is lower.')