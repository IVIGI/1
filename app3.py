
from requests import get

list1 = input('Введите Номер 1-150 = ')
list2 = input('Введите вид пингвина 0-2 = ')
list3 = input('Введите остров пингвина 0-2 = ')
list4 = input('Введите длину пингвина от 35.0 до 45.0 = ')
list5 = input('Введите глубину клюва пингвина от 16.0 до 23.0 = ')
list6 = input('Введите длину ласта пингвина от 170 до 200 = ')
list7 = input('Введите вес в гр 3700 - 4500 = ')
list8 = input('Введите год измерения данных 2004-2007 = ')
print(get(f'http://localhost:5000/api?list1={list1}&list2={list2}&list3={list3}&list4={list4}&list5={list5}&list6={list6}&list7={list7}&list8={list8}').json())