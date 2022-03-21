clear;
%% Part 1
% 2-2-1 neural network for XOR problem
x1 = [-1 -1 1 1];
x2 = [-1 1 -1 1];
t = [-1 1 1 -1];
eta = 0.1;
theta = 0.001;

% function f(.)
fx = @(x) tanh(x);
dfx = @(x) 1 - fx(x)^2;

dimension = [2 2 1];

% structure of weights
%{
|  bias   |
| input 1 |
| input 2 |
%}
wij = [0.6 -1 0.2;
    -1.2 -0.55 1.6]';
wkj = [1.3 -0.8 0.42]';

% hidden unit output
nety1 = @(n) wij(1,1) + wij(2, 1) * x1(n) + wij(3, 1) * x2(n);
nety2 = @(n) wij(1,2) + wij(2,2) * x1(n) + wij(2,3) * x2(n);

% final output
netz = @(i) wkj(1) + wkj(2) * fx(nety1(i)) + wkj(3) * fx(nety2(i));

max_epoch = 300;

for r = 1:max_epoch % epoch counter
    delw_ij = [0 0 0;
        0 0 0]'; % delta accumulators (for batch)

    delw_jk = [0; 0; 0];

    % m = which column we're targeting for the sample input and target output
    for m = 1:length(x1) 
        xm = [1; x1(m); x2(m)];
        y = [1; fx(nety1(m)); fx(nety2(m))];
        
        zk = fx(netz(m));
        delk = (t(m) - zk) * dfx(wkj' * y);
        for i = 1:length(delw_ij)
            
            
            for j = 1:width(delw_ij)
                
                delw_ij(i, j) = delw_jk(i, j) + (eta * xm(i) * dfx(wij(:, j)' * xm) * sum(wkj * delk));
            end
        end

        for j = 1:length(delw_jk)
            delw_jk(j) = delw_jk(j) + (eta * (t(j) - zk) * dfx(netz(j)) * y(j));
        end

        %disp([delw_jk'; delw_ij']);

    end

    wij = wij + delw_ij;
    wkj = wkj + delw_jk;

    z = fx(netz(1:4));
    J(r, :) = 0.5 * (t - z).^2;

    if (r > 1)
        if (sum(J(r)) - sum(J(r-1)) < theta)
            break;
        end
    end
end


% Plot the Learning Curve
figure;
hold all;
n = [0:1:length(J)-1];
plot(n,J)
grid;
title('Learning Curve For J(r)');
ylabel('J(r)');
xlabel('r');
hold off;


figure;
x11=(-3:3);
y1 = -(wij(2, 1)/wij(3, 1)) * x11 - (wij(1, 1)/wij(3, 1));
plot(x11,y1,'--');

w01=w12(1);
wji=w12(2);
wkj=w12(3);
x22=-(wji/wkj)*x11-(w01/wkj);

hold on;

boundedline=plot(x11,x22,'-');

for i=1:length(x1)
    if (ak(i)<0)
        false=plot(x1(i),x2(i),'bo');
    else
        true=plot(x1(i),x2(i),'bx');
    end
end


hold on; 
grid;

title('x1 vs. x2 Decision Boundaries');
xlabel('x1');
ylabel('x2');

legend([true,false,boundedline],'XOR True = 1','XOR False = -1 ','Boundary Line');


correct=0;
accuracy=0;


for i=1:length(x1)
    if floor(ak(i))==t(i) || ceil(ak(i))==t(i)
        correct=correct+1;
    end
end

accuracy=correct*100/length(x1);
accuracy;



y11=(-3:3);
w01=w21(1);
wji=w21(2);
wkj=w21(3);
y21=-(wji/wkj)*x11-(w01/wkj);

plot(x11,x22,'k');

hold on; grid;

title('Decision Space - y1 vs y2');

xlabel('y1');
ylabel('y2');
