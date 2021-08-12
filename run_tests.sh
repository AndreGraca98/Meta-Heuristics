echo 'Running exercise 1'
python ex1.py -n 30
python ex1.py -n 128
python ex1.py -n 312 #-i 2e4


echo 'Running exercise 2.1'
python ex2.py -e 1 -c 1e-2
python ex2.py -e 1 -c 1e-3
python ex2.py -e 1 -c 1e-4
python ex2.py -e 1 -c 1e-5

echo 'Running exercise 2.2'
python ex2.py -e 2 -c 1e-2
python ex2.py -e 2 -c 1e-3
