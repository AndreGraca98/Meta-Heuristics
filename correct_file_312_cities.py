import os

os.chdir('cio')



if __name__ == '__main__':
    with open('CIO_TSP_312_2021.txt') as f:
        original = f.readlines()
        original = original[3:]
        # print(original[0])

    arr = []
    for s in original:
        arr.append(s.strip('\n'))

    lines = '''#  N x N distance matrix.
    #  N = 312
    #\n'''
    counter = 0
    for o in arr:
        if counter == 32:
            lines += '\n'
            counter = 0
        lines += o
        counter += 1

    with open('CIO_TSP_312_2021_correct.txt', 'w') as f:
        f.write(lines)
        