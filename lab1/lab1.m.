%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posteriors_x,g_x]=lab1(x,Training_Data, feature)

% x = individual sample to be tested (to identify its probable class label)
% feature = index of relevant feature (column) in Training_Data (value of 1-4)
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
f=D(:, feature);
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% prior probability is just the total prob of setosa/versicolour
disp('Prior probabilities:');
Pr1 = length(find(la == 1))/length(la);
Pr2 = length(find(la == 2))/length(la);
disp([num2str(Pr1), ' ', num2str(Pr2)])

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Mean & Std for class 1 & 2');
m11 = mean(f(1:50));  % mean of the class conditional density p(x/w1)
std11 = std(f(1:50)); % Standard deviation of the class conditional density p(x/w1)

m12 = mean(f(51:100)); % mean of the class conditional density p(x/w2)
std12= std(f(51:100)); % Standard deviation of the class conditional density p(x/w2)

disp(['Class 1 Mean: ', num2str(m11), ' std: ', num2str(std11)]);
disp(['Class 2 Mean: ', num2str(m12), ' std: ', num2str(std12)]);

disp(['Conditional probabilities for x=' num2str(x)]);
% calculate p(x/w1)
cp11 = (1/(sqrt(2*pi) * std11)) * exp((-1/2) * ((x - m11)/std11).^2); 
 % calculate p(x/w2)
cp12 = (1/(sqrt(2*pi) * std12)) * exp((-1/2) * ((x - m12)/std12).^2); 

disp(['Conditional Probabilities class 1: ', num2str(cp11), ' class 2: ', num2str(cp12)]);

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

disp('Posterior prob. for the test feature');
% p(w1/x) denominator = 1 since we only have 2 classes, so all samples will
% fall under either w1 or w2
pos11 = cp11 * Pr1 / 1;
% p(w2/x)
pos12 = cp12 * Pr2 / 1;

posteriors_x = [pos11 pos12];

disp(['Posteriors: ', num2str(posteriors_x)]);

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

% compute the g(x) for min err rate classifier.
g_x = pos11 - pos12;

if g_x > 0
    disp("Sample classified as Iris Setosa");
else
    disp("Sample classified as Iris Versicolour");
end
