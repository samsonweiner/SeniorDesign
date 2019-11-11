#test file to run linear regression and get some sort of visual
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import csv

presPath = "Datasets/Opioid_prescription_amounts.csv"
popPath = "Datasets/County_Population_2010-2018.csv"

def readPrescriptions():
    x_row, y_row = [], []
    with open(presPath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if str(row[0][0]) == "0" and str(row[0][1]) == "6":
                while len(row) != 1:
                    row[0] += row[1]
                    row.pop(1)
                    #print(row)
            temprow = row[0].split(",")
            #print(temprow)
            if len(temprow) >= 5 and "CA" in temprow:
                x_row.append(temprow[0][1:])
                y_row.append(temprow[4])
               # print(temprow[0], temprow[4])
            #print(', '.join(row))
    return x_row[1:], y_row[1:]

def plotPrescription():
    x, y = readPrescriptions()
    N = 1000
    colors = (0, 0, 0)
    area = np.pi
    for i in range(len(x)):
        x[i] = float(x[i])
        y[i] = float(y[i])

    plt.ylim = (6000, 6125)
    plt.figure(figsize=(12,8))
    plt.scatter(x, y, s=area, c=colors, alpha=1.0)
    plt.title('Prescription Amounts by Zip Code in CA')
    plt.xlabel('Zip Code')
    plt.ylabel('Prescription Amounts')
    plt.show()

def readPopulation():
    x_row, y_row = [], []
    with open(popPath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        hit = 0
        for row in spamreader:
            temprow = str(', '.join(row))
            if 'County' in temprow and "California" in temprow:
                hit = 1
                count = 0
                idx = temprow.index('County')
                temprow = temprow[:idx + 7] + temprow[idx + 8:]
                temprow = temprow.split(",")
                x_row.append(temprow[1])
                y_row.append(temprow[12])
            else:
                count = 1
                if count == 1 and hit == 1:
                    break
    return x_row, y_row

def plotPopulation():
    x, y = readPopulation()
    N = 1000
    colors = (0, 0, 0)
    area = np.pi
    for i in range(len(x)):
        x[i] = float(x[i])
        y[i] = float(y[i])

    plt.ylim = (0, 10050000)
    plt.figure(figsize=(12,8))
    plt.scatter(x, y, s=area, c=colors, alpha=1.0)
    
    plt.title('Population by Zip Code in CA')
    plt.xlabel('Zip Code')
    plt.ylabel('Population')
    plt.show()

def plotLinear():
    pres_x, pres_y = readPrescriptions()
    pop_x, pop_y = readPopulation()
    dataDict = {}
    for i in range(len(pres_x)):
        dataDict[pres_x[i]] = [pres_y[i]]

    for i in range(len(pop_x)):
        if pop_x[i] in dataDict:
            dataDict[pop_x[i]].append(pop_y[i])
    
    dat_x, dat_y = [], []
    for x in dataDict:
        dat_x.append(float(dataDict[x][0]))
        dat_y.append(float(dataDict[x][1]))
    
    colors = (0, 0, 0)
    area = np.pi

    val = 10000000

    linSlope, linInt, r_value, p_value, std_err = stats.linregress(dat_y, dat_x)
    t_x = [0, val]
    t_y = [linInt, val*linSlope + linInt]

    plt.ylim = (0, 10050000)
    plt.figure(figsize=(12,8))
    plt.plot(t_x, t_y)
    plt.scatter(dat_y, dat_x, s=area, c=colors, alpha=1.0)
    plt.title('Population vs Prescription per Cap')
    plt.xlabel('Population')
    plt.ylabel('Prescriptions')
    plt.show()



def main():
    graph = input('Which graph to run? (prescription, population, linear)')
    if str(graph) == 'prescription':
        plotPrescription()
    if str(graph) == 'population':
        plotPopulation()
    if str(graph) == 'linear':
        plotLinear()

if __name__ == '__main__':
    main()