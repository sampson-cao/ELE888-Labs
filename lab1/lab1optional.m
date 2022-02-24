%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONAL PART
function [g_x]=lab1optional(x,Training_Data, feature1, feature2)

% x = individual sample to be tested (to identify its probable class label)
% feature1, feature 2 = index of relevant feature (column) in Training_Data (value of 1-4)
% Training_Data = Matrix containing the training samples and numeric class labels
% posterior_x  = Posterior probabilities
% g_x = value of the discriminant function

D=Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[M,N]=size(D);    

% Feature samples
%   1 - Sepal Length
%   2 - Sepal Width
%   3 - Petal Length
%   4 - Petal Width
f=D(:, [feature1, feature2]);
la=D(:,N); % class labels

disp('--------------------------------');
disp('Sample Provided');
disp(x);

%%%%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% prior probability is just the total prob of setosa/versicolour
disp('Prior probabilities:');
Pr1 = length(find(la == 1))/length(la);
Pr2 = length(find(la == 2))/length(la);
disp([num2str(Pr1), ' ', num2str(Pr2)])

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Means & Standard Deviations');
% Mean and std of feature 1 class 1
m11 = mean(f(1:50, 1));
std11 = std(f(1:50, 1));

% Mean and std of feature 1 class 2
m12 = mean(f(51:100, 1));
std12= std(f(51:100, 1));

% mean and std of feature 2 class 1
m21 = mean(f(1:50, 2));
std21 = std(f(1:50, 2));

% mean and std of feature 2 class 2
m22 = mean(f(51:100, 2));
std22 = std(f(51:100, 2));

disp(['Feature 1 Class 1 Mean: ', num2str(m11), ' std: ', num2str(std11)]);
disp(['Feature 1 Class 2 Mean: ', num2str(m12), ' std: ', num2str(std12)]);
disp(['Feature 2 Class 1 Mean: ', num2str(m21), ' std: ', num2str(std21)]);
disp(['Feature 2 Class 2 Mean: ', num2str(m22), ' std: ', num2str(std22)]);

% Construct the covariance matrix for class 1
sig112 = 1/50 * sum((f(1:50, 1) - m11) * (f(1:50, 2) - m21).', 'all');

sig1 = [std11 sig112; 
    sig112 std21];

% Construct the covariance matrix for class 2
sig212 = 1/50 * sum((f(51:100, 1) - m12) * (f(51:100, 2) - m22).', 'all');

sig2 = [std12 sig212; 
    sig212 std22];

% calculate p(x/w1)
cp11 = 1/(2*pi * sqrt(det(sig1))) * exp((-1/2) * ((x - [m11; m21]).' * inv(sig1) * (x - [m11; m21]))); 
% calculate p(x/w2)
cp12 = 1/(2*pi * sqrt(det(sig2))) * exp((-1/2) * ((x - [m12; m22]).' * inv(sig2) * (x - [m12; m22]))); 

disp('Conditional Probabilities class 1: ');
disp(cp11);
disp('Conditional Probabilities class 2: ');
disp(cp12);
%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

disp('Posterior prob. for the test feature');

% Posterior using conditional probability
pos11 = cp11 * Pr1 / 1;
pos12 = cp12 * Pr2 / 1;

% Classification using covariance matrix
%{
pos11 = (x.' * (-1/2 * inv(sig1)) * x) ...
    + ((inv(sig1) * [m11; m21]).' * x) ...
    + ((-1/2 * [m11 m21] * inv(sig1) * [m11; m21])) ...
    - (1/2 * log(det(sig1)) + log(Pr1));

pos12 = (x.' * (-1/2 * inv(sig2)) * x) ... 
    + ((inv(sig2) * [m12; m22]).' * x) ...
    + ((-1/2 * [m12 m22] * inv(sig2) * [m12; m22])) ...
    - (1/2 * log(det(sig2)) + log(Pr2));
%}

posteriors_x = [pos11 pos12];

disp('Posteriors: ');
disp(posteriors_x);

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

% compute the g(x) for min err rate classifier.
g_x = pos11 - pos12;

if g_x > 0
    disp("Sample classified as Iris Setosa");
else
    disp("Sample classified as Iris Versicolour");
end

disp(g_x);
