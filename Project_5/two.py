import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def evalFun( x , rbar , theta, T, rt ):
    
    # First term of objective function - q1
    q1 = - np.sum( rbar * x )

    # Second term of objective function - q2
    inner_bracket = np.sum((rt-rbar) * x , axis=1)
    q2 = (theta/T) * np.sum(inner_bracket ** 2 , axis=0)

    return q1 + q2

def evalGrad(x, rbar, theta, T, rt):

    # First term of gradient
    q1 = -rbar

    # Second term of gradient
    q2 = ((theta * 2) / T) * np.sum( (( (rt - rbar) * x )).T @ (rt-rbar), axis=0)
    # q2 = ((theta * 2) / T) * np.sum( (np.multiply( (rt - rbar) , x )).T @ (rt-rbar), axis=0)

    return q1 + q2


def run_grad_descent_momentum(x, rbar, theta, T, rt, xsol, fvalsol, N,  mu, loudsteps):
        
        converged = False 
        olddeltas = np.zeros_like(x)
        np.copyto(olddeltas, x)
        
        # chunk_size = 2500
        al = 0.01
        chunk_size = 5000

        num_workers = 10 # Giving each worker chunk_size assets for gradient computation
        p = Pool(num_workers)

        for iteration in range(N):

            np.copyto( xsol[iteration], x)

            args = [(x[i:min(i+chunk_size, len(x))], rbar[i:min(i+chunk_size, len(x))], theta, T, rt[:, i:min(i+chunk_size, len(x))] ) for i in range(0, len(x), chunk_size)]

            # Splitting up gradient computation 
            gradResults = p.starmap_async(evalGrad, args)

            fvalsol[iteration] = evalFun(x, rbar, theta, T, rt)

            g_n = np.concatenate(gradResults.get())
            g_n /= np.linalg.norm(g_n)
            
            if loudsteps or 0 == iteration%10:
                print('\nIteration', iteration)
                print('at function value = {:.4e} '.format(fvalsol[iteration]))

            deltas = -g_n*mu + (1-mu)*olddeltas

            # Projected Gradient
            deltas[x >= 1] = np.minimum(0, deltas[x>=1])
            deltas[x <= -1] = np.maximum(0, deltas[x<=-1])

            newx = x + al*deltas

            np.copyto(olddeltas, deltas)

            ## Constraint on xj
            newx = np.minimum(1,newx)
            newx = np.maximum(-1,newx)

            np.copyto(x, newx)

            al *= 0.98

            if (iteration>1) and abs(fvalsol[iteration] - fvalsol[iteration-1]) < 1e-6:
            # if abs(fval - newfval) < 1e-8:
                converged = True
                break 
        p.close()
        p.join()
                
        # print('\n*** Done at iteration {} and converged {} with final gradient ( {}, {}, {})\n'.format(iteration,converged, g0, g1, g2))
        return iteration, converged


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: one.py N T \u03B8 num_asset_names")
        print("N = max number of iterations for GD")
        print("T = number of days of returns data")
        print("\u03B8 = theta for objective function")
        print("num_asset_names = Number of stocks from which pairs will be formed")

        print("This implementation is for asset names greater than 200")

        sys.exit("Bye!")

    # Initialize the parameters from command line argument
    N = int(sys.argv[1])
    T = int(sys.argv[2])
    theta = float(sys.argv[3])
    num_assets = int(sys.argv[4])

    combined_returns = pd.read_csv("./Returns.csv")
    combined_returns = combined_returns.iloc[ : T  , : num_assets]

    combined_returns_cols_arr = np.array(combined_returns.columns)
    combined_returns_arr = np.array(combined_returns, dtype=np.float16)

    # Create a list of pairs indices
    i, j = np.triu_indices(num_assets, k=1)
    all_pairs_indices = np.column_stack((i, j))

    num_pairs = (num_assets*(num_assets-1))//2
    print(f"Taking {num_assets} assets, resulting in {num_pairs} pairs.")

    print("Starting the creation of pair returns i.e. deltas")
    deltas = (combined_returns_arr[ : , all_pairs_indices[:, 0]] - combined_returns_arr[ : , all_pairs_indices[:, 1]])
    deltas = deltas.astype(np.float16)
    print("Deltas created successfully!")

    print("Creating delta averages ....")
    delta_bar = np.mean(deltas, axis=0)
    print("Delta averages done!")

    x = np.random.uniform(-1, 1, size=num_pairs).astype(np.float32)
    x /= np.sum(np.abs(x))

    xsol = np.zeros(( N , num_pairs ), dtype=np.float32)
    fvalsol = np.zeros( N , dtype=np.float32)

    print("Starting GD")
    iteration, converged = run_grad_descent_momentum(x, delta_bar, theta, T, deltas, xsol, fvalsol, N, 0.995, False)

    print(f"Stopped at Iteration {iteration}, Converged = {converged}")

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # Plot convergence of function value against iteration
    axes[0].scatter(np.arange(0, iteration + 1), fvalsol[:iteration + 1], marker=".")
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Function Value')
    axes[0].set_title(f'\u03B8={theta}, T={T}, assets={num_assets}, pairs={num_pairs} at fval={np.round(fvalsol[iteration],4):.4f}')

    # Plot portfolio created
    axes[1].bar(np.arange(1, num_pairs+1), xsol[iteration])
    axes[1].set_xlabel('Pair')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Portfolio Distribution')

    plt.show()