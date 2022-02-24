%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 1, Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    1. sepal length in cm
%    2. sepal width in cm
%    3. petal length in cm
%    4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3


%% this script will run lab1 experiments..
clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% feature distribution of x1 for two classes
figure

    
subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),2),100), title('Iris Setosa, sepal width (cm)');
subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),2),100); title('Iris Veriscolour, sepal width (cm)');

figure

subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),1),100), title('Iris Setosa, sepal length (cm)');
subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),1),100); title('Iris Veriscolour, sepal length (cm)');
    

figure

plot(irisdata_features(find(numericLabels(:)==1),1),irisdata_features(find(numericLabels(:)==1),2),'rs'); title('x_1 vs x_2');
hold on;
plot(irisdata_features(find(numericLabels(:)==2),1),irisdata_features(find(numericLabels(:)==2),2),'k.');
axis([4 7 1 5]);

    

%% build training data set for two class comparison
% merge feature samples with numeric labels for two class comparison (Iris
% Setosa vs. Iris Veriscolour
trainingSet = [irisdata_features(1:100,:) numericLabels(1:100,1) ];


%% Lab1 experiments (include here)
input = [3.3 4.4 5 5.7 6.3];
output = [];
for in = input
    output = [output, lab1(in, trainingSet, 2)];
    disp(' ');
end

% 4. The decision boundary appears to be x = 3.1 and x = -0.29 (setosa
% transitions to versicolour at and vice versa

% 5. If there was a higher penalty for classifying w2 than w1, then the
% system would preferentially classify x as w1 over w2. This would lead to
% the decision boundary being shifted towards w1, and a higher
% rate of misclasssifcation of x as w1 over w2
x = [0:0.05:6];
y = 1/(2*pi*0.37719)*exp(-(x-3.418).^2/(2*0.37719^2));
figure();
a1 = plot (x, y);
b1 = "Setosa";
hold on;
y = 1/(2*pi*0.31064)*exp(-(x-2.77).^2/(2*0.31064^2));
a2 = plot (x, y);
b2 = "Versicolour";
hold off;
legend ([a1, a2], [b1, b2]);
title("Plot of PDF of Setosa and Versicolour");
ylabel("Conditional Probability p(\omega)");
xlabel("Sepal Width x (cm)");

output2 = [];
for in = input
    output2 = [output2, lab1(in, trainingSet, 1)];
    disp(' ');
end

% Based on the mean and std, it seems like w2 is a better feature

%% Lab 1 Optional Part
clc;
input2 = [[2 6]; [4.4 3]; [5 3.5]; [5.3 2]; 
          [5.5 2.5]; [6.6 3.5]; [4.5 6.1]];

output2 = [];
for in = input2.'
    output2 = [output2, lab1optional(in, trainingSet, 1, 2)];
    disp(' ');
end

x = 1:0.1:9;
y = 1:0.1:9;
z = zeros(length(x));
[X, Y] = meshgrid(x, y);

disp('Part 2 Optional');
for i = 1:length(x)
    for j = 1:length(y)
        z(i, j) = abs(lab1optional([x(i); y(j)], trainingSet, 1, 2));
    end
end

figure();
mesh(X, Y, z);

%{
[2 6] - Setosa
[4.4 3] - Setosa
[5 3.5] - Setosa
[5.3 2] - Versicolour
[5.5 2.5] - Versicolour
[6.6 3.5] - Versicolour
[4.5 6.1] - Setosa
%}
