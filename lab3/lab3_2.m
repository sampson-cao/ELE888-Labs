clear;
%% Part 2
% 2-2-1 neural network for wine dataset

% Shape training data so that class 3 is negative
wineData = load('wine.data');
TrainingSet(1:59,:,:) = wineData(1:59,1:3); %Class 1=w1
TrainingSet(60:107,:,:) = wineData(131:178,1:3); %Class 3=w2
for i=60:length(TrainingSet)
    TrainingSet(i,1)=-1;
end

t = TrainingSet(:, 1)';
z = zeros(1, length(t));
x1 = TrainingSet(:,2)'; 
x2 = TrainingSet(:,3)'; 

% Normalize dataset
mean_x1 = mean(x1);
std_x1 = std(x1);
Norm_x1 = (x1 - mean_x1)/std_x1;
mean_x2 = mean(x2);
std_x2 = std(x2);
Norm_x2 = (x2 - mean_x2)/std_x2;
x1 = Norm_x1;
x2 = Norm_x2;

eta = 0.1;
theta = 0.001;

% function f(.)
fx = @(x) tanh(x);
dfx = @(x) sech(x)^2;

dimension = [2 2 1];

% structure of weights
%{
|  bias   |
| input 1 |
| input 2 |
%}
% Random weights used to initialize model
wij = [0.807 0.9 -1;
    -1.53 -0.9 -1]';
wkj = [1 0.5 0.6]';


% final output
% netz = @(i) wkj(1) + wkj(2) * fx(nety1(i)) + wkj(3) * fx(nety2(i));

max_epoch = 300;

for r = 1:max_epoch % epoch counter
    delw_ij = [0 0 0;
        0 0 0]'; % delta accumulators (for batch)

    delw_jk = [0; 0; 0];

    % m = which column we're targeting for the sample input and target output
    for m = 1:length(x1)
        xm = [1; x1(m); x2(m)];
        y = [1; fx(wij(:, 1)' * xm); fx(wij(:, 2)' * xm)];
        netk = wkj' * y;
        zk = fx(netk);
        delk = (t(m) - zk) * dfx(wkj' * y);
        
        for j = 1:width(delw_ij)
            delj(j) = dfx(wij(:, j)' * xm) * wkj(j+1) * delk;
        end
        

        delw_ij = delw_ij + (eta * xm * delj);
        delw_jk = delw_jk + (eta * delk * y);

        z(m) = zk;

    end

    wij = wij + delw_ij
    wkj = wkj + delw_jk;

    J(r) = 0.5 * norm(t - z)^2;

    if (r > 1)
        if (abs(J(r) - J(r-1)) < theta)
            disp('Error is below theta')
            break;
        end
    end
end

disp('Epochs: ');
disp(r);


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


figure();
x11=(-3:3);
y1 = -(wij(2, 1)/wij(3, 1)) * x11 - (wij(1, 1)/wij(3, 1));
plot(x11,y1,'--');

y2 = -(wij(2, 2)/wij(3, 2)) * x11 - (wij(1, 2)/wij(3, 2));

hold on;

boundedline=plot(x11,y2,'-');

for i=1:length(x1)
    if t(i) < 0
        false=plot(x1(i),x2(i),'bo');
    else
        true=plot(x1(i),x2(i),'bx');
    end
end
hold on;
grid;

title('x1 vs. x2 Decision Boundaries');
xlabel('Malic Acid');
ylabel('Alcohol Content');

legend([true,false,boundedline],'XOR True = 1','XOR False = -1 ','Boundary Line');


correct=0;
accuracy=0;

for i=1:length(x1)
    if floor(z(i))==t(i) || ceil(z(i))==t(i)
        correct=correct+1;
    end
end

accuracy=correct*100/length(x1);
accuracy