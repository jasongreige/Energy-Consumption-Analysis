import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def convert_kwh_to_mmbtu(value):
    return value * 3412 / 1000000

def compare_energetic_intensity(myIntensity, referenceIntensity):
    if myIntensity > referenceIntensity :
        print('Mon bâtiment est plus efficace que le bâtiment de référence.\n')
    elif myIntensity < referenceIntensity:
        print('Mon bâtiment est moins efficace que le bâtiment de référence.\n')
    else:
        print('Mon bâtiment est également efficace au bâtiment de référence.\n')

if __name__ == '__main__':
    #QUESTION 1:
    file_path = 'Copie de Input.xlsx'
    df = pd.read_excel(file_path)
    pd.set_option('display.max_columns', None) #display all columns without truncation
    pd.set_option('display.width', 1000)  # set the display width to a high value
    df = df.round(2)
    print(df)

    #QUESTION 2:
    building_area = 110000
    reference_energetic_intensity = 100

    total_natural_gas = df.loc[df['Month'] == 'Total', 'Natural Gas, MMBtu'].values[0] * 1000
    total_electric = convert_kwh_to_mmbtu(df.loc[df['Month'] == 'Total', 'Electric, kWh'].values[0]) * 1000
    total_energetic_intensity = (total_electric + total_natural_gas) / building_area

    print('\nl’intensité énergétique totale du bâtiment est de: ' + str(total_energetic_intensity) + ' kBtu/pi^2\n')
    compare_energetic_intensity(total_energetic_intensity, reference_energetic_intensity)

    #QUESTION 1:

    df['Month'] = df['Month'].apply(lambda x: x[:3] if x != "Total" else x)
    df['Natural Gas'] = df['Natural Gas, MMBtu']
    df['Electric'] = convert_kwh_to_mmbtu(df['Electric, kWh'])

    ax = df.plot(x='Month', y=['Natural Gas', 'Electric'], kind='bar', rot=0, width=0.8, figsize=(10, 6))

    ax.set_title('Consommation énergétique par mois')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Consommation énergétique (MMBtu)')

    ax.grid(axis='y', linestyle='--')


    #QUESTION 3 et 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    df['DJR'] = df.iloc[:-1]['CDD, °F*Day'] #ignore the Total row
    ax1.scatter(df['DJR'], df['Electric'])
    ax1.set_xlabel('DJR, °F*Jour')
    ax1.set_ylabel('Électricité, MMBtu')
    ax1.set_title('Electricité par rapport aux DJR')

    df['DJC'] = df.iloc[:-1]['HDD, °F*Day'] #ignore the Total row
    ax2.scatter(df['DJC'], df['Natural Gas'])
    ax2.set_xlabel('DJC, °F*Jour')
    ax2.set_ylabel('Gaz Naturel, MMBtu')
    ax2.set_title('Gaz naturel par rapport aux DJC')


    df = df.dropna(subset=['DJR', 'Electric'])
    djr_array = np.array(df['DJR']).reshape((-1, 1))
    electric_array = np.array(df['Electric'])

    linearRegression_electric = LinearRegression()
    linearRegression_electric.fit(djr_array, electric_array)

    predicition_electric = linearRegression_electric.predict(djr_array)
    equation_electric = 'y = {:.2f} * x + {:.2f}'.format(linearRegression_electric.coef_[0], linearRegression_electric.intercept_)
    r_squared_electric = r2_score(electric_array, predicition_electric)

    ax1.plot(df['DJR'], predicition_electric, color='red')
    ax1.annotate(equation_electric, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    ax1.text(0.05, 0.85, f"R^2 = {r_squared_electric:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')

    print('Pente de la droite du nuage de points de l`électricité: ', linearRegression_electric.coef_[0])
    print('Ordonnée de la droite du nuage de de l`électricité: ', linearRegression_electric.intercept_)


    df = df.dropna(subset=['DJC', 'Natural Gas'])
    djc_array = np.array(df['DJC']).reshape((-1, 1))
    nat_gas_array = np.array(df['Natural Gas'])

    linearRegression_natural_gas = LinearRegression()
    linearRegression_natural_gas.fit(djc_array, nat_gas_array)

    prediction_nat_gas = linearRegression_natural_gas.predict(djc_array)
    equation_nat_gas = 'y = {:.2f} * x + {:.2f}'.format(linearRegression_natural_gas.coef_[0], linearRegression_natural_gas.intercept_)
    r_squared_natural_gas = r2_score(nat_gas_array, prediction_nat_gas)

    ax2.plot(df['DJC'], prediction_nat_gas, color='red')
    ax2.annotate(equation_nat_gas, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    ax2.text(0.05, 0.85, f"R^2 = {r_squared_natural_gas:.2f}", transform=ax2.transAxes, fontsize=12, verticalalignment='top')

    print('Pente de la droite du nuage de points du gas naturel: ', linearRegression_natural_gas.coef_[0])
    print('Ordonnée de la droite du nuage de points du gas naturel: ', linearRegression_natural_gas.intercept_)

    plt.show()

