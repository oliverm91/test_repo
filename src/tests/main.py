from matplotlib import pyplot as plt
import numpy as np

from my_package.my_funcs import no_sub_package_return_1
from my_package.curve_models import NelsonSiegelSvenson

if __name__ == '__main__':
    resultado1 = no_sub_package_return_1()
    assert resultado1==1, f'no_sub_package_return_1 debería dar 1, pero fue {resultado1}.'

    print('All tests passed')

    tenors_list = [1, 94, 184, 276, 368, 549, 733, 1098, 1463, 1829, 2194, 2559, 2924, 3290, 3655, 4385, 5481, 7307]
    tenors = np.array(tenors_list)/365
    dfs_list = [0.999739793752311, 0.977399123023781, 0.958671154543347, 0.941169991274706, 0.925651968346516, 0.897844863653134, 0.872874239623946, 0.825253951997447, 0.780536376005809, 0.737931329587201, 0.695565023515513, 0.656365940336681, 0.619894961429865, 0.584245213672534, 0.54809468187623, 0.483451073576347, 0.40128984492367, 0.295415512387573]
    dfs = np.array(dfs_list)

    zcrs = -np.log(dfs)/tenors
    nsv = NelsonSiegelSvenson(tenors, zcrs)
    zcrs_model = nsv.get_zero_rates(tenors)

    # Plotting the line for the first set of y values
    plt.plot(tenors, zcrs, label='Original Curve', marker='o')

    # Plotting the line for the second set of y values
    plt.plot(tenors, zcrs_model, label='Model Curve', marker='o')

    # Adding labels and title
    plt.xlabel('Tenors Y')
    plt.ylabel('Zero Rates')
    plt.title('Zero rates comparison')

    # Displaying legend
    plt.legend()

    # Display the plot
    plt.show()