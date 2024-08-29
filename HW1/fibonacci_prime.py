import sys

def is_prime(n):
    if n < 2:
        return False

    flag = True
    test = 2
    while flag and test < n**0.5:
        if n % test == 0:
            flag = False
        test += 1

    return flag


def get_fibonacci(iter_max=100,check_prime=True):
    series = [0,1]
    i = 0
    while i < iter_max:
        series.append(series[-2]+series[-1])
        i += 1

    for i in range(len(series)):
        if check_prime:
            print(f'{i}:',series[i],{True: 'is prime', False: 'is not prime'}[is_prime(series[i])])
        else:
            print(f'{i}:',series[i])



if __name__ == '__main__':
    iter_max = 10
    if len(sys.argv) > 1:
        iter_max = int(sys.argv[1])
    get_fibonacci(iter_max=iter_max) 
