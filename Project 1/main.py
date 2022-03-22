import numpy as np
import scipy.io
import math
import geneNewData

# GLOBAL VARS

# prior probability
p = 0.5

def main():
    # Christine Nguyen
    # Student ID: 1217604687
    myID='4687'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')


    print('executing task 1 . . .')
    arr_bright_digit0, arr_std_digit0 = extract_features(train0);
    arr_bright_digit1, arr_std_digit1 = extract_features(train1);

    print('executing task 2 . . .')
    mean_bright_0, var_bright_0, mean_std_digit0, var_std_digit0= calc_params(arr_bright_digit0, arr_std_digit0)
    mean_bright_1, var_bright_1, mean_std_digit1, var_std_digit1 = calc_params(arr_bright_digit1, arr_std_digit1)

    print('0 f0 mean: ', mean_bright_0)
    print('0 f0 variance: ', var_bright_0)
    print('0 f1 mean: ', mean_std_digit0)
    print('0 f1 variance: ', var_std_digit0)
    
    print('1 f0 mean: ', mean_bright_1)
    print('1 f0 variance: ', var_bright_1)
    print('1 f1 mean: ', mean_std_digit1)
    print('1 f1 variance: ', var_std_digit1)

    print('executing task 3 . . .')
    output_digit0, output_digit1 = build_naive_bayesian_model(test0, test1, mean_bright_0, var_bright_0, mean_std_digit0, var_std_digit0, mean_bright_1, var_bright_1, mean_std_digit1, var_std_digit1)

    print('executing task 4 . . .')
    accuracy_digit0, accuracy_digit1 = calc_accuracy(output_digit0, output_digit1, test0, test1)
    print('accuracy digit 0: ', accuracy_digit0)
    print('accuracy digit 1: ', accuracy_digit1)



# task 1: extract features from original dataset + convert original data arrays into 2d data points
def extract_features(trainset):
    # 2 features: average brightness + standard deviation of brightness of each image
    # assumptions:
    # - 2 features are independent
    # - each image drawn from normal distribution

    arr_bright = []
    arr_std = []

    for image in trainset:
        arr_bright.append(np.average(image))
        arr_std.append(np.std(image))
    
    return arr_bright, arr_std


def calc_params(avg_brightness, std): 
    f0_mean = np.mean(avg_brightness)
    f0_variance = np.var(avg_brightness)

    f1_mean = np.mean(std)
    f1_variance = np.var(std)

    return f0_mean, f0_variance, f1_mean, f1_variance


# naive bayes formula components
# f0 = brightness 
# f1 = std
def build_naive_bayesian_model(digit0testset, digit1testset, mean_f0_digit0, var_f0_digit0, mean_f1_digit0, var_f1_digit0, mean_f0_digit1, var_f0_digit1, mean_f1_digit1, var_f1_digit1):
    arr_bright_digit0test = []
    arr_std_digit0test = []
    arr_bright_digit1test = []
    arr_std_digit1test = []
    output_digit0 = []
    output_digit1 = []

    arr_bright_digit0test, arr_std_digit0test = extract_features(digit0testset)
    arr_bright_digit1test, arr_std_digit1test = extract_features(digit1testset)
    

    # digit 0
    for i in range(len(arr_bright_digit0test)):
        pdf_a = calc_pdf(arr_bright_digit0test[i], mean_f0_digit0, np.sqrt(var_f0_digit0))
        pdf_b = calc_pdf(arr_std_digit0test[i], mean_f1_digit0, np.sqrt(var_f1_digit0))
        post_prob_ab = pdf_a * pdf_b * p

        pdf_c = calc_pdf(arr_bright_digit0test[i], mean_f0_digit1, np.sqrt(var_f0_digit1));
        pdf_d = calc_pdf(arr_std_digit0test[i], mean_f1_digit1, np.sqrt(var_f1_digit1));
        post_prob_cd= pdf_c * pdf_d * p

        if post_prob_ab >= post_prob_cd:
            output_digit0.append(0)
        else:
            output_digit0.append(1)

    # digit 1
    for i in range(len(arr_bright_digit1test)):
        pdf_a = calc_pdf(arr_bright_digit1test[i], mean_f0_digit0, np.sqrt(var_f0_digit0))
        pdf_b = calc_pdf(arr_std_digit1test[i],  mean_f1_digit1, np.sqrt(var_f1_digit1))
        post_prob_ab = pdf_a * pdf_b * p

        pdf_c = calc_pdf(arr_bright_digit1test[i], mean_f0_digit1, np.sqrt(var_f0_digit1));
        pdf_d = calc_pdf(arr_std_digit1test[i], mean_f1_digit1, np.sqrt(var_f1_digit1));
        post_prob_cd= pdf_c * pdf_d * p

        if post_prob_cd >= post_prob_ab:
            output_digit1.append(1)
        else:
            output_digit1.append(0)

    
    return output_digit0, output_digit1

# calcuate probability density function
# where x = value of the array
# mu = mean
# sigma = std        
def calc_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- ((x - mu)**2) / (2 * (sigma)**2))


def calc_accuracy(post_prob_0, post_prob_1, digit0testset, digit1testset):
    accuracy_digit0 = 0
    accuracy_digit1 = 0
    values_0 = 0
    values_1 = 0

    for value in post_prob_0:
        if value == 0:
            values_0 += 1
    
    for value in post_prob_1:
        if value == 1:
            values_1 += 1

    accuracy_digit0 = values_0 / len(digit0testset)
    accuracy_digit1 = values_1 / len(digit1testset)
    
    return accuracy_digit0, accuracy_digit1



if __name__ == '__main__':
    main()